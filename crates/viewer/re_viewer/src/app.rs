use std::ops::Bound;
use std::str::FromStr as _;
use std::sync::Arc;
use std::{iter::once, time::Duration};

use ahash::HashMap;
use egui::{FocusDirection, Key};
use itertools::Itertools as _;
use re_build_info::CrateVersion;
use re_capabilities::MainThreadToken;
use re_chunk::{TimeInt, TimelineName};
use re_data_source::{AuthErrorHandler, FileContents, LogDataSource};
use re_entity_db::InstancePath;
use re_entity_db::entity_db::EntityDb;
use re_log_channel::{
    DataSourceMessage, DataSourceUiCommand, LogReceiver, LogReceiverSet, LogSource,
};
use re_log_types::{
    ApplicationId, FileSource, LogMsg, RecordingId, StoreId, StoreKind, TableMsg, TimeReal,
};
use re_redap_client::ConnectionRegistryHandle;
use re_renderer::{ScreenshotProcessor, WgpuResourcePoolStatistics};
use re_sdk_types::blueprint::components::{LoopMode, PlayState};
use re_ui::egui_ext::context_ext::ContextExt as _;
use re_ui::{ContextExt as _, UICommand, UICommandSender as _, UiExt as _, notifications};
use re_view_map::MapView;
use re_view_spatial::{SpatialView2D, SpatialView3D};
use re_viewer_context::open_url::{OpenUrlOptions, ViewerOpenUrl, combine_with_base_url};
use re_viewer_context::store_hub::{BlueprintPersistence, StoreHub, StoreHubStats};
use re_viewer_context::{
    AppOptions, AsyncRuntimeHandle, AuthContext, BlueprintUndoState, CommandReceiver,
    CommandSender, ComponentUiRegistry, DisplayMode, EditRedapServerModalCommand,
    FallbackProviderRegistry, Item, NeedsRepaint, PublishedViewInfo, RecordingOrTable,
    StorageContext, StoreContext, SystemCommand, SystemCommandSender as _, TableStore,
    TimeControlCommand, ViewClass, ViewClassRegistry, ViewClassRegistryError, ViewId,
    ViewRectPublisher, command_channel, gpu_bridge, sanitize_file_name,
};
use re_viewport::ViewportUi;
use re_viewport_blueprint::ViewportBlueprint;

use crate::AppState;
use crate::app_blueprint::{AppBlueprint, PanelStateOverrides};
use crate::app_blueprint_ctx::AppBlueprintCtx;
use crate::app_state::{WelcomeScreenState, active_view_ids_for_export};
use crate::background_tasks::BackgroundTasks;
use crate::event::ViewerEventDispatcher;
use crate::startup_options::StartupOptions;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write as _;
#[cfg(not(target_arch = "wasm32"))]
use std::process::Stdio;

#[cfg(not(target_arch = "wasm32"))]
const OFFSCREEN_EXPORT_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

// ----------------------------------------------------------------------------

/// Storage key used to store the last run Rerun version.
///
/// This is then used to detect if the user has recently upgraded Rerun.
const RERUN_VERSION_KEY: &str = "rerun.version";

const REDAP_TOKEN_KEY: &str = "rerun.redap_token";

#[cfg(not(target_arch = "wasm32"))]
const MIN_ZOOM_FACTOR: f32 = 0.2;
#[cfg(not(target_arch = "wasm32"))]
const MAX_ZOOM_FACTOR: f32 = 5.0;

#[cfg(target_arch = "wasm32")]
struct PendingFilePromise {
    recommended_store_id: Option<StoreId>,
    force_store_info: bool,
    promise: poll_promise::Promise<Vec<re_data_source::FileContents>>,
}

#[cfg(not(target_arch = "wasm32"))]
const CROPPED_VIDEO_EXPORT_PROMISE: &str = "cropped_video_export";

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CroppedVideoExportPhase {
    AwaitRenderedFrame,
    AwaitViewScreenshots,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
struct VideoExportView {
    view_id: ViewId,
    rect: egui::Rect,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub(crate) struct VideoExportArea {
    name: String,
    rect: egui::Rect,
    views: Vec<VideoExportView>,
    pixels_per_point: f32,
}

#[cfg(not(target_arch = "wasm32"))]
struct CapturedVideoViewFrame {
    image: image::RgbaImage,
    ui_rect: egui::Rect,
    pixels_per_point: f32,
}

#[cfg(not(target_arch = "wasm32"))]
struct CroppedVideoExport {
    export_id: u64,
    store_id: StoreId,
    original_timeline: TimelineName,
    original_time: Option<TimeReal>,
    original_play_state: PlayState,
    export_area: VideoExportArea,
    output_path: std::path::PathBuf,
    temp_dir: std::path::PathBuf,
    frame_times: Vec<TimeInt>,
    frame_index: usize,
    frames_until_capture: u8,
    phase: CroppedVideoExportPhase,
    fps: f32,
    captured_views: HashMap<ViewId, CapturedVideoViewFrame>,
}

#[cfg(not(target_arch = "wasm32"))]
struct OffscreenViewportRenderer {
    egui_ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_format: wgpu::TextureFormat,
}

/// The Rerun Viewer as an [`eframe`] application.
pub struct App {
    #[allow(clippy::allow_attributes, dead_code)] // Unused on wasm32
    main_thread_token: MainThreadToken,
    build_info: re_build_info::BuildInfo,

    app_env: crate::AppEnvironment,

    startup_options: StartupOptions,
    start_time: web_time::Instant,
    ram_limit_warner: re_memory::RamLimitWarner,
    pub(crate) egui_ctx: egui::Context,
    screenshotter: crate::screenshotter::Screenshotter,

    #[cfg(target_arch = "wasm32")]
    pub(crate) popstate_listener: Option<crate::web_history::PopstateListener>,

    #[cfg(not(target_arch = "wasm32"))]
    profiler: re_tracing::Profiler,

    /// Listens to the local text log stream
    text_log_rx: std::sync::mpsc::Receiver<re_log::LogMsg>,

    component_ui_registry: ComponentUiRegistry,
    component_fallback_registry: FallbackProviderRegistry,

    rx_log: LogReceiverSet,

    #[cfg(target_arch = "wasm32")]
    open_files_promise: Option<PendingFilePromise>,

    #[cfg(not(target_arch = "wasm32"))]
    cropped_video_export: Option<CroppedVideoExport>,

    #[cfg(not(target_arch = "wasm32"))]
    next_cropped_video_export_id: u64,

    #[cfg(not(target_arch = "wasm32"))]
    renderer_video_export_state: Arc<gpu_bridge::RendererVideoExportState>,

    /// What is serialized
    pub(crate) state: AppState,

    /// Pending background tasks, e.g. files being saved.
    pub(crate) background_tasks: BackgroundTasks,

    /// Interface for all recordings and blueprints
    pub(crate) store_hub: Option<StoreHub>,

    /// Notification panel.
    pub(crate) notifications: notifications::NotificationUi,

    memory_panel: crate::memory_panel::MemoryPanel,
    memory_panel_open: bool,

    egui_debug_panel_open: bool,

    /// Last time the latency was deemed interesting.
    ///
    /// Note that initializing with an "old" `Instant` won't work reliably cross platform
    /// since `Instant`'s counter may start at program start.
    pub(crate) latest_latency_interest: Option<web_time::Instant>,

    /// Measures how long a frame takes to paint
    pub(crate) frame_time_history: egui::util::History<f32>,

    /// Commands to run at the end of the frame.
    pub command_sender: CommandSender,
    command_receiver: CommandReceiver,
    cmd_palette: re_ui::CommandPalette,

    /// All known view types.
    view_class_registry: ViewClassRegistry,

    pub(crate) panel_state_overrides_active: bool,
    pub(crate) panel_state_overrides: PanelStateOverrides,

    reflection: re_types_core::reflection::Reflection,

    /// External interactions with the Viewer host (JS, custom egui app, notebook, etc.).
    pub event_dispatcher: Option<ViewerEventDispatcher>,

    connection_registry: ConnectionRegistryHandle,

    /// The async runtime that should be used for all asynchronous operations.
    ///
    /// Using the global tokio runtime should be avoided since:
    /// * we don't have a tokio runtime on web
    /// * we want the user to have full control over the runtime,
    ///   and not expect that a global runtime exists.
    async_runtime: AsyncRuntimeHandle,
}

impl App {
    pub fn new(
        main_thread_token: MainThreadToken,
        build_info: re_build_info::BuildInfo,
        app_env: crate::AppEnvironment,
        startup_options: StartupOptions,
        creation_context: &eframe::CreationContext<'_>,
        connection_registry: Option<ConnectionRegistryHandle>,
        tokio_runtime: AsyncRuntimeHandle,
    ) -> Self {
        Self::with_commands(
            main_thread_token,
            build_info,
            app_env,
            startup_options,
            creation_context,
            connection_registry,
            tokio_runtime,
            crate::register_text_log_receiver(),
            command_channel(),
        )
    }

    /// Create a viewer that receives new log messages over time
    #[expect(clippy::too_many_arguments)]
    pub fn with_commands(
        main_thread_token: MainThreadToken,
        build_info: re_build_info::BuildInfo,
        app_env: crate::AppEnvironment,
        startup_options: StartupOptions,
        creation_context: &eframe::CreationContext<'_>,
        connection_registry: Option<ConnectionRegistryHandle>,
        tokio_runtime: AsyncRuntimeHandle,
        text_log_rx: std::sync::mpsc::Receiver<re_log::LogMsg>,
        command_channel: (CommandSender, CommandReceiver),
    ) -> Self {
        re_tracing::profile_function!();

        {
            let command_sender = command_channel.0.clone();
            re_auth::credentials::subscribe_auth_changes(move |user| {
                command_sender.send_system(SystemCommand::OnAuthChanged(
                    user.map(|user| AuthContext { email: user.email }),
                ));
            });
        }

        let connection_registry = connection_registry
            .unwrap_or_else(re_redap_client::ConnectionRegistry::new_with_stored_credentials);

        if let Some(storage) = creation_context.storage
            && let Some(tokens) = eframe::get_value(storage, REDAP_TOKEN_KEY)
        {
            connection_registry.load_tokens(tokens);
        }

        let mut state: AppState = if startup_options.persist_state {
            creation_context.storage
                .and_then(|storage| {
                    // This re-implements: `eframe::get_value` so we can customize the warning message.
                    // TODO(#2849): More thorough error-handling.
                    let value = storage.get_string(eframe::APP_KEY)?;
                    match ron::from_str(&value) {
                        Ok(value) => Some(value),
                        Err(err) => {
                            re_log::warn!("Failed to restore application state. This is expected if you have just upgraded Rerun versions.");
                            re_log::debug!("Failed to decode RON for app state: {err}");
                            None
                        }
                    }
                })
                .unwrap_or_default()
        } else {
            AppState::default()
        };

        if startup_options.persist_state {
            // Check if the user has recently upgraded Rerun.
            if let Some(storage) = creation_context.storage {
                let current_version = build_info.version;
                let previous_version: Option<CrateVersion> =
                    storage.get_string(RERUN_VERSION_KEY).and_then(|version| {
                        // `CrateVersion::try_parse` is `const` (for good reasons), and needs a `&'static str`.
                        // In order to accomplish this, we need to leak the string here.
                        let version = Box::leak(version.into_boxed_str());
                        CrateVersion::try_parse(version).ok()
                    });

                if previous_version
                    .is_none_or(|previous_version| previous_version < CrateVersion::new(0, 24, 0))
                {
                    re_log::debug!(
                        "Upgrading from {} to {}.",
                        previous_version.map_or_else(|| "<unknown>".to_owned(), |v| v.to_string()),
                        current_version
                    );
                    // We used to have Dark as the hard-coded theme preference. Let's change that!
                    creation_context
                        .egui_ctx
                        .options_mut(|o| o.theme_preference = egui::ThemePreference::System);
                }
            }
        }

        if let Some(video_decoder_hw_acceleration) = startup_options.video_decoder_hw_acceleration {
            state.app_options.video_decoder_hw_acceleration = video_decoder_hw_acceleration;
        }

        if app_env.is_test() {
            // Disable certain labels/warnings/etc that would be flaky or not CI-runner-agnostic in snapshot tests.
            state.app_options.show_metrics = false;
        }

        let mut component_fallback_registry =
            re_component_fallbacks::create_component_fallback_registry();

        let view_class_registry =
            crate::default_views::create_view_class_registry(&mut component_fallback_registry)
                .unwrap_or_else(|err| {
                    re_log::error!("Failed to create view class registry: {err}");
                    Default::default()
                });

        #[allow(clippy::allow_attributes, unused_mut, clippy::needless_update)]
        // false positive on web
        let mut screenshotter = crate::screenshotter::Screenshotter::default();

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(screenshot_path) = startup_options.screenshot_to_path_then_quit.clone() {
            screenshotter.screenshot_to_path_then_quit(&creation_context.egui_ctx, screenshot_path);
        }

        let (command_sender, command_receiver) = command_channel;

        #[cfg(not(target_arch = "wasm32"))]
        let renderer_video_export_state = Arc::new(gpu_bridge::RendererVideoExportState::default());

        let mut component_ui_registry = re_component_ui::create_component_ui_registry();
        re_data_ui::register_component_uis(&mut component_ui_registry);

        let (_adapter_backend, _device_tier) = creation_context.wgpu_render_state.as_ref().map_or(
            (
                wgpu::Backend::Noop,
                re_renderer::device_caps::DeviceCapabilityTier::Limited,
            ),
            |render_state| {
                let mut egui_renderer = render_state.renderer.write();
                if egui_renderer
                    .callback_resources
                    .get::<Arc<gpu_bridge::RendererVideoExportState>>()
                    .is_none()
                {
                    egui_renderer
                        .callback_resources
                        .insert(renderer_video_export_state.clone());
                }
                let render_ctx = egui_renderer
                    .callback_resources
                    .get::<re_renderer::RenderContext>();

                (
                    render_state.adapter.get_info().backend,
                    render_ctx.map_or(
                        re_renderer::device_caps::DeviceCapabilityTier::Limited,
                        |ctx| ctx.device_caps().tier,
                    ),
                )
            },
        );

        #[cfg(feature = "analytics")]
        if let Some(analytics) = re_analytics::Analytics::global_or_init() {
            use crate::viewer_analytics::event;

            analytics.record(event::identify(
                analytics.config(),
                build_info.clone(),
                &app_env,
            ));
            analytics.record(event::viewer_started(
                &app_env,
                &creation_context.egui_ctx,
                _adapter_backend,
                _device_tier,
            ));
        }

        let panel_state_overrides = startup_options.panel_state_overrides;

        let reflection = re_sdk_types::reflection::generate_reflection().unwrap_or_else(|err| {
            re_log::error!(
                "Failed to create list of serialized default values for components: {err}"
            );
            Default::default()
        });

        let event_dispatcher = startup_options
            .on_event
            .clone()
            .map(ViewerEventDispatcher::new);

        if !state.redap_servers.is_empty() {
            command_sender.send_ui(UICommand::ExpandBlueprintPanel);
        }

        creation_context.egui_ctx.on_end_pass(
            "remove copied text formatting",
            Arc::new(|ctx| {
                ctx.output_mut(|o| {
                    for command in &mut o.commands {
                        if let egui::output::OutputCommand::CopyText(text) = command {
                            *text = re_format::remove_number_formatting(text);
                        }
                    }
                });
            }),
        );

        {
            // TODO(emilk/egui#7659): This is a workaround consuming the Space/Arrow keys so we can
            // use them as timeline shortcuts. Egui's built in behavior is to interact with focus,
            // and we don't want that.
            // But of course text edits should still get it so we use this ugly hack to check if
            // a text edit is focused.
            let command_sender = command_sender.clone();
            creation_context.egui_ctx.on_begin_pass(
                "filter space key",
                Arc::new(move |ctx| {
                    if !ctx.text_edit_focused() {
                        let conflicting_commands = [
                            UICommand::PlaybackTogglePlayPause,
                            UICommand::PlaybackBeginning,
                            UICommand::PlaybackEnd,
                            UICommand::PlaybackForwardFast,
                            UICommand::PlaybackBackFast,
                            UICommand::PlaybackStepForward,
                            UICommand::PlaybackStepBack,
                            UICommand::PlaybackForward,
                            UICommand::PlaybackBack,
                        ];

                        let os = ctx.os();
                        let mut reset_focus_direction = false;
                        ctx.input_mut(|i| {
                            for command in conflicting_commands {
                                for shortcut in command.kb_shortcuts(os) {
                                    if i.consume_shortcut(&shortcut) {
                                        if shortcut.logical_key == Key::ArrowLeft
                                            || shortcut.logical_key == Key::ArrowRight
                                        {
                                            reset_focus_direction = true;
                                        }
                                        command_sender.send_ui(command);
                                    }
                                }
                            }
                        });

                        if reset_focus_direction {
                            // Additionally, we need to revert the focus direction on ArrowLeft/Right
                            // keys to prevent the focus change for timeline shortcuts
                            ctx.memory_mut(|mem| {
                                mem.move_focus(FocusDirection::None);
                            });
                        }
                    }
                }),
            );
        }

        Self {
            main_thread_token,
            build_info,
            app_env,
            startup_options,
            start_time: web_time::Instant::now(),
            ram_limit_warner: re_memory::RamLimitWarner::warn_at_fraction_of_max(0.75),
            egui_ctx: creation_context.egui_ctx.clone(),
            screenshotter,

            #[cfg(target_arch = "wasm32")]
            popstate_listener: None,

            #[cfg(not(target_arch = "wasm32"))]
            profiler: Default::default(),

            text_log_rx,
            component_ui_registry,
            component_fallback_registry,
            rx_log: Default::default(),

            #[cfg(target_arch = "wasm32")]
            open_files_promise: Default::default(),

            #[cfg(not(target_arch = "wasm32"))]
            cropped_video_export: None,

            #[cfg(not(target_arch = "wasm32"))]
            next_cropped_video_export_id: 1,

            #[cfg(not(target_arch = "wasm32"))]
            renderer_video_export_state,

            state,
            background_tasks: Default::default(),
            store_hub: Some(StoreHub::new(
                blueprint_loader(),
                &crate::app_blueprint::setup_welcome_screen_blueprint,
            )),
            notifications: notifications::NotificationUi::new(creation_context.egui_ctx.clone()),

            memory_panel: Default::default(),
            memory_panel_open: false,

            egui_debug_panel_open: false,

            latest_latency_interest: None,

            frame_time_history: egui::util::History::new(1..100, 0.5),

            command_sender,
            command_receiver,
            cmd_palette: Default::default(),

            view_class_registry,

            panel_state_overrides_active: true,
            panel_state_overrides,

            reflection,

            event_dispatcher,

            connection_registry,
            async_runtime: tokio_runtime,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_profiler(&mut self, profiler: re_tracing::Profiler) {
        self.profiler = profiler;
    }

    pub fn connection_registry(&self) -> &ConnectionRegistryHandle {
        &self.connection_registry
    }

    pub fn set_examples_manifest_url(&mut self, url: String) {
        re_log::info!("Using manifest_url={url:?}");
        self.state.set_examples_manifest_url(&self.egui_ctx, url);
    }

    pub fn build_info(&self) -> &re_build_info::BuildInfo {
        &self.build_info
    }

    pub fn startup_options(&self) -> &StartupOptions {
        &self.startup_options
    }

    pub fn app_options(&self) -> &AppOptions {
        self.state.app_options()
    }

    pub fn app_options_mut(&mut self) -> &mut AppOptions {
        self.state.app_options_mut()
    }

    pub fn app_env(&self) -> &crate::AppEnvironment {
        &self.app_env
    }

    /// Open a content URL in the viewer.
    pub fn open_url_or_file(&self, url: &str) {
        match ViewerOpenUrl::from_str(url) {
            Ok(url) => {
                url.open(
                    &self.egui_ctx,
                    &OpenUrlOptions {
                        follow_if_http: false,
                        select_redap_source_when_loaded: true,
                        show_loader: true,
                    },
                    &self.command_sender,
                );
            }
            Err(err) => {
                if err.to_string().contains(url) {
                    re_log::error!("{err}");
                } else {
                    re_log::error!("Failed to open URL {url}: {err}");
                }
            }
        }
    }

    pub fn is_screenshotting(&self) -> bool {
        self.screenshotter.is_screenshotting()
    }

    #[expect(clippy::needless_pass_by_ref_mut)]
    pub fn add_log_receiver(&mut self, rx: LogReceiver) {
        re_log::debug!("Adding new log receiver: {}", rx.source());

        // Make sure we wake up when a new message is available:
        rx.set_waker({
            let egui_ctx = self.egui_ctx.clone();
            move || {
                // Spend a few more milliseconds decoding incoming messages,
                // then trigger a repaint (https://github.com/rerun-io/rerun/issues/963):
                egui_ctx.request_repaint_after(std::time::Duration::from_millis(10));
            }
        });

        // Add unknown redap servers.
        //
        // Otherwise we end up in a situation where we have a data from an unknown server,
        // which is unnecessary and can get us into a strange ui state.
        if let LogSource::RedapGrpcStream { uri, .. } = rx.source() {
            self.command_sender
                .send_system(SystemCommand::AddRedapServer(uri.origin.clone()));
        }

        self.rx_log.add(rx);
    }

    /// Update the active [`re_viewer_context::TimeControl`]. And if the blueprint inspection
    /// panel is open, also open that time control.
    fn move_time(&mut self) {
        if let Some(store_hub) = &self.store_hub
            && let Some(store_id) = store_hub.active_store_id()
            && let Some(blueprint) = store_hub.active_blueprint_for_app(store_id.application_id())
        {
            let default_blueprint = store_hub.default_blueprint_for_app(store_id.application_id());

            let blueprint_query = self
                .state
                .get_blueprint_query_for_viewer(blueprint)
                .unwrap_or_else(|| {
                    re_chunk::LatestAtQuery::latest(re_viewer_context::blueprint_timeline())
                });

            let bp_ctx = AppBlueprintCtx {
                command_sender: &self.command_sender,
                current_blueprint: blueprint,
                default_blueprint,
                blueprint_query,
            };

            let dt = self.egui_ctx.input(|i| i.stable_dt);
            if let Some(recording) = store_hub.active_recording() {
                // Are we still connected to the data source for the current store?
                let more_data_is_coming =
                    recording.data_source.as_ref().is_some_and(|store_source| {
                        self.rx_log
                            .sources()
                            .iter()
                            .any(|s| s.as_ref() == store_source)
                    });

                let time_ctrl = self.state.time_control_mut(recording, &bp_ctx);

                // The state diffs are used to trigger callbacks if they are configured.
                // If there's no active recording, we should not trigger any callbacks, but since there's an active recording here,
                // we want to diff state changes.
                let should_diff_state = true;
                let response = time_ctrl.update(
                    recording.timeline_histograms(),
                    dt,
                    more_data_is_coming,
                    should_diff_state,
                    Some(&bp_ctx),
                );

                if response.needs_repaint == NeedsRepaint::Yes {
                    self.egui_ctx.request_repaint();
                }

                handle_time_ctrl_event(recording, self.event_dispatcher.as_ref(), &response);
            }

            if self.app_options().inspect_blueprint_timeline {
                let more_data_is_coming = true;
                let should_diff_state = false;
                // We ignore most things from the time control response for the blueprint but still
                // need to repaint if requested.
                let re_viewer_context::TimeControlResponse {
                    needs_repaint,
                    playing_change: _,
                    timeline_change: _,
                    time_change: _,
                } = self.state.blueprint_time_control.update(
                    bp_ctx.current_blueprint.timeline_histograms(),
                    dt,
                    more_data_is_coming,
                    should_diff_state,
                    None::<&AppBlueprintCtx<'_>>,
                );

                if needs_repaint == NeedsRepaint::Yes {
                    self.egui_ctx.request_repaint();
                }

                let undo_state = self
                    .state
                    .blueprint_undo_state
                    .entry(blueprint.store_id().clone())
                    .or_default();
                // Apply changes to the blueprint time to the undo-state:
                if self.state.blueprint_time_control.play_state() == PlayState::Following {
                    undo_state.redo_all();
                } else if let Some(time) = self.state.blueprint_time_control.time_int() {
                    undo_state.set_redo_time(time);
                }
            }
        }
    }

    pub fn msg_receive_set(&self) -> &LogReceiverSet {
        &self.rx_log
    }

    /// Adds a new view class to the viewer.
    pub fn add_view_class<T: ViewClass + Default + 'static>(
        &mut self,
    ) -> Result<(), ViewClassRegistryError> {
        self.view_class_registry
            .add_class::<T>(&mut self.component_fallback_registry)
    }

    /// Accesses the view class registry which can be used to extend the Viewer.
    ///
    /// **WARNING:** Many parts or the viewer assume that all views & visualizers are registered before the first frame is rendered.
    /// Doing so later in the application life cycle may cause unexpected behavior.
    pub fn view_class_registry(&mut self) -> &mut ViewClassRegistry {
        &mut self.view_class_registry
    }

    pub fn component_fallback_registry(&mut self) -> &mut FallbackProviderRegistry {
        &mut self.component_fallback_registry
    }

    fn check_keyboard_shortcuts(&self, egui_ctx: &egui::Context) {
        if let Some(cmd) = UICommand::listen_for_kb_shortcut(egui_ctx) {
            self.command_sender.send_ui(cmd);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn is_cropped_video_busy(&self) -> bool {
        self.cropped_video_export.is_some()
            || self
                .background_tasks
                .is_promise_in_progress(CROPPED_VIDEO_EXPORT_PROMISE)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn video_export_area(
        &self,
        store_context: &StoreContext<'_>,
    ) -> Option<VideoExportArea> {
        let pixels_per_point = self.egui_ctx.pixels_per_point();
        let blueprint_query = self
            .state
            .get_blueprint_query_for_viewer(store_context.blueprint)?;
        let viewport = re_viewport_blueprint::ViewportBlueprint::from_db(
            store_context.blueprint,
            &blueprint_query,
        );

        self.egui_ctx.memory_mut(|mem| {
            let view_rects = mem.caches.cache::<ViewRectPublisher>();
            let visible_views = viewport
                .view_ids()
                .filter_map(|view_id| {
                    let view = viewport.view(view_id)?;
                    let info = view_rects.get(view_id)?.clone();
                    Some((*view_id, view.class_identifier(), info))
                })
                .collect_vec();

            if visible_views.is_empty()
                || visible_views.iter().any(|(_, class_identifier, _)| {
                    !is_renderer_video_export_supported(*class_identifier)
                })
            {
                return None;
            }

            let visible_view_layouts = visible_views
                .iter()
                .map(|(view_id, _, info)| VideoExportView {
                    view_id: *view_id,
                    rect: info.rect,
                })
                .collect_vec();

            let mut rects = visible_view_layouts.iter().map(|info| info.rect);
            let mut rect = rects.next()?;
            for other in rects {
                rect = rect.union(other);
            }
            rect = rect.shrink(2.5);

            if !rect.is_positive() {
                return None;
            }

            let name = match visible_views
                .iter()
                .map(|(_, _, info)| info)
                .collect_vec()
                .as_slice()
            {
                [PublishedViewInfo { name, .. }] => name.clone(),
                _ => "viewport".to_owned(),
            };

            Some(VideoExportArea {
                name,
                rect,
                views: visible_view_layouts,
                pixels_per_point,
            })
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn restore_after_cropped_video_export(&self, export: &CroppedVideoExport) {
        let mut time_commands = vec![TimeControlCommand::SetActiveTimeline(
            export.original_timeline,
        )];
        if let Some(time) = export.original_time {
            time_commands.push(TimeControlCommand::SetTime(time));
        }
        time_commands.push(TimeControlCommand::SetPlayState(export.original_play_state));

        self.command_sender
            .send_system(SystemCommand::TimeControlCommands {
                store_id: export.store_id.clone(),
                time_commands,
            });
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn abort_cropped_video_export(&mut self, reason: anyhow::Error) {
        self.renderer_video_export_state.clear_request();
        let Some(export) = self.cropped_video_export.take() else {
            re_log::error!("{reason}");
            return;
        };

        self.restore_after_cropped_video_export(&export);
        if let Err(err) = std::fs::remove_dir_all(&export.temp_dir)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            re_log::debug!(
                "Failed to clean up temporary video frames at {:?}: {err}",
                export.temp_dir
            );
        }

        re_log::error!("{reason}");
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn finish_cropped_video_capture(&mut self) {
        self.renderer_video_export_state.clear_request();
        let Some(export) = self.cropped_video_export.take() else {
            return;
        };

        self.restore_after_cropped_video_export(&export);

        let output_path = export.output_path.clone();
        let temp_dir = export.temp_dir.clone();
        let temp_dir_for_encoding = temp_dir.clone();
        let fps = export.fps;

        if let Err(err) = self
            .background_tasks
            .spawn_threaded_promise(CROPPED_VIDEO_EXPORT_PROMISE, move || {
                encode_cropped_video_frames(&temp_dir_for_encoding, &output_path, fps)
            })
        {
            if let Err(cleanup_err) = std::fs::remove_dir_all(&temp_dir)
                && cleanup_err.kind() != std::io::ErrorKind::NotFound
            {
                re_log::debug!(
                    "Failed to clean up temporary video frames at {:?}: {cleanup_err}",
                    temp_dir
                );
            }
            re_log::error!("Failed to start cropped video export: {err}");
            return;
        }

        re_log::info!(
            "Captured {} frame(s) for {:?}; encoding MP4 in the background.",
            export.frame_times.len(),
            export.export_area.name
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn prepare_cropped_video_export_frame_request(&mut self) -> Option<(u64, usize, Vec<ViewId>)> {
        let export = self.cropped_video_export.as_mut()?;

        match export.phase {
            CroppedVideoExportPhase::AwaitRenderedFrame => {
                if export.frames_until_capture > 0 {
                    export.frames_until_capture -= 1;
                    self.egui_ctx.request_repaint();
                    None
                } else {
                    export.phase = CroppedVideoExportPhase::AwaitViewScreenshots;
                    Some((
                        export.export_id,
                        export.frame_index,
                        export
                            .export_area
                            .views
                            .iter()
                            .map(|view| view.view_id)
                            .collect(),
                    ))
                }
            }
            CroppedVideoExportPhase::AwaitViewScreenshots => {
                self.egui_ctx.request_repaint();
                None
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn collect_cropped_video_view_frames(&mut self, render_ctx: &re_renderer::RenderContext) {
        let Some(export_id) = self
            .cropped_video_export
            .as_ref()
            .map(|export| export.export_id)
        else {
            return;
        };

        let mut readbacks = Vec::new();
        while ScreenshotProcessor::next_readback_result(
            render_ctx,
            export_id,
            |data, extent, screenshot: gpu_bridge::RendererVideoExportScreenshot| {
                readbacks.push((data.to_vec(), extent, screenshot));
            },
        )
        .is_some()
        {}

        for (data, extent, screenshot) in readbacks {
            if let Err(err) = self.handle_cropped_video_view_frame(&data, extent, screenshot) {
                self.abort_cropped_video_export(err);
                break;
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn handle_cropped_video_view_frame(
        &mut self,
        data: &[u8],
        extent: glam::UVec2,
        screenshot: gpu_bridge::RendererVideoExportScreenshot,
    ) -> anyhow::Result<()> {
        enum NextStep {
            QueueNextFrame {
                store_id: StoreId,
                next_time: TimeReal,
            },
            Finish,
        }

        let Some(export) = self.cropped_video_export.as_mut() else {
            return Ok(());
        };

        if export.export_id != screenshot.export_id || export.frame_index != screenshot.frame_index
        {
            return Ok(());
        }

        let Some(view_image) = image::RgbaImage::from_raw(extent.x, extent.y, data.to_vec()) else {
            anyhow::bail!(
                "Failed to reconstruct renderer screenshot for {:?}",
                screenshot.view_id
            );
        };
        export.captured_views.insert(
            screenshot.view_id,
            CapturedVideoViewFrame {
                image: view_image,
                ui_rect: screenshot.ui_rect,
                pixels_per_point: screenshot.pixels_per_point,
            },
        );

        if export.captured_views.len() < export.export_area.views.len() {
            self.egui_ctx.request_repaint();
            return Ok(());
        }

        let composited =
            composite_cropped_video_frame(export, self.egui_ctx.style().visuals.panel_fill)?;
        write_cropped_video_frame_rgba(&export.temp_dir, screenshot.frame_index, &composited)?;
        export.captured_views.clear();

        let next_step = if screenshot.frame_index + 1 < export.frame_times.len() {
            export.frame_index += 1;
            export.frames_until_capture = 1;
            export.phase = CroppedVideoExportPhase::AwaitRenderedFrame;

            NextStep::QueueNextFrame {
                store_id: export.store_id.clone(),
                next_time: export.frame_times[export.frame_index].into(),
            }
        } else {
            NextStep::Finish
        };

        match next_step {
            NextStep::QueueNextFrame {
                store_id,
                next_time,
            } => {
                self.command_sender
                    .send_system(SystemCommand::TimeControlCommands {
                        store_id,
                        time_commands: vec![TimeControlCommand::SetTime(next_time)],
                    });
                self.egui_ctx.request_repaint();
            }
            NextStep::Finish => {
                self.finish_cropped_video_capture();
                self.egui_ctx.request_repaint();
            }
        }

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_offscreen_cropped_video_export(
        &mut self,
        frame_render_state: &egui_wgpu::RenderState,
        store_context: &StoreContext<'_>,
        storage_context: &StorageContext<'_>,
    ) {
        let Some(export) = self.cropped_video_export.take() else {
            return;
        };

        match self.run_offscreen_cropped_video_export_inner(
            frame_render_state,
            store_context,
            storage_context,
            export,
        ) {
            Ok((output_path, frame_count, name)) => {
                re_log::info!(
                    "Saved {} frame(s) for {:?} to {:?}.",
                    frame_count,
                    name,
                    output_path
                );
            }
            Err(err) => {
                re_log::error!("{err}");
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_offscreen_cropped_video_export_inner(
        &mut self,
        frame_render_state: &egui_wgpu::RenderState,
        store_context: &StoreContext<'_>,
        storage_context: &StorageContext<'_>,
        export: CroppedVideoExport,
    ) -> anyhow::Result<(std::path::PathBuf, usize, String)> {
        let mut offscreen_renderer =
            OffscreenViewportRenderer::new(frame_render_state, &self.egui_ctx)?;

        let blueprint_query = self
            .state
            .blueprint_query_for_viewer(store_context.blueprint);
        let viewport_blueprint =
            ViewportBlueprint::from_db(store_context.blueprint, &blueprint_query);
        let viewport_ui = ViewportUi::new(viewport_blueprint);
        if viewport_ui.blueprint.is_invalid() {
            anyhow::bail!("No valid viewport layout is available for video export");
        }
        let active_view_ids = active_view_ids_for_export(&viewport_ui.blueprint);
        let visualizable_entities_per_visualizer = self
            .view_class_registry
            .visualizable_entities_for_visualizer_systems(store_context.recording.store_id());
        let indicated_entities_per_visualizer = self
            .view_class_registry
            .indicated_entities_per_visualizer(store_context.recording.store_id());

        let app_blueprint_ctx = AppBlueprintCtx {
            command_sender: &self.command_sender,
            current_blueprint: store_context.blueprint,
            default_blueprint: store_context.default_blueprint,
            blueprint_query: blueprint_query.clone(),
        };

        let mut export_time_controls = HashMap::default();
        let export_time_ctrl = crate::app_state::create_time_control_for(
            &mut export_time_controls,
            store_context.recording,
            &app_blueprint_ctx,
        );
        let _ = export_time_ctrl.handle_time_commands(
            None::<&AppBlueprintCtx<'_>>,
            store_context.recording.timeline_histograms(),
            &[
                TimeControlCommand::SetActiveTimeline(export.original_timeline),
                TimeControlCommand::SetPlayState(PlayState::Paused),
            ],
        );

        let frame_width = ((export.export_area.rect.width() * export.export_area.pixels_per_point)
            .round() as u32)
            .max(1);
        let frame_height =
            ((export.export_area.rect.height() * export.export_area.pixels_per_point).round()
                as u32)
                .max(1);
        let mut ffmpeg = spawn_streaming_video_encoder(
            &export.output_path,
            export.fps,
            frame_width,
            frame_height,
        )?;

        let temp_dir = export.temp_dir.clone();
        let result = (|| -> anyhow::Result<(std::path::PathBuf, usize, String)> {
            for frame_time in export.frame_times.iter().copied() {
                let _ = export_time_ctrl.handle_time_commands(
                    None::<&AppBlueprintCtx<'_>>,
                    store_context.recording.timeline_histograms(),
                    &[TimeControlCommand::SetTime(frame_time.into())],
                );

                let image = offscreen_renderer.render_viewport_frame(
                    export.export_area.rect.size(),
                    export.export_area.pixels_per_point,
                    |ctx, render_ctx| {
                        egui::CentralPanel::default()
                            .frame(egui::Frame::NONE)
                            .show(ctx, |ui| {
                                self.state.render_prepared_viewport_only(
                                    &self.app_env,
                                    ui,
                                    render_ctx,
                                    store_context,
                                    storage_context,
                                    &self.reflection,
                                    &self.component_ui_registry,
                                    &self.component_fallback_registry,
                                    &self.view_class_registry,
                                    &self.rx_log,
                                    &self.command_sender,
                                    &self.connection_registry,
                                    &blueprint_query,
                                    &viewport_ui,
                                    &active_view_ids,
                                    &visualizable_entities_per_visualizer,
                                    &indicated_entities_per_visualizer,
                                    export_time_ctrl,
                                );
                            });
                    },
                )?;

                write_video_frame_to_ffmpeg(&mut ffmpeg, &image)?;
            }

            finish_streaming_video_encoder(ffmpeg, &export.output_path)?;

            Ok((
                export.output_path,
                export.frame_times.len(),
                export.export_area.name,
            ))
        })();

        if let Err(err) = std::fs::remove_dir_all(&temp_dir)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            re_log::debug!(
                "Failed to clean up temporary video frames at {:?}: {err}",
                temp_dir
            );
        }

        result
    }

    fn run_pending_system_commands(&mut self, store_hub: &mut StoreHub, egui_ctx: &egui::Context) {
        while let Some((from_where, cmd)) = self.command_receiver.recv_system() {
            self.run_system_command(from_where, cmd, store_hub, egui_ctx);
        }
    }

    fn run_pending_ui_commands(
        &mut self,
        egui_ctx: &egui::Context,
        app_blueprint: &AppBlueprint<'_>,
        storage_context: &StorageContext<'_>,
        store_context: Option<&StoreContext<'_>>,
        display_mode: &DisplayMode,
    ) {
        while let Some(cmd) = self.command_receiver.recv_ui() {
            self.run_ui_command(
                egui_ctx,
                app_blueprint,
                storage_context,
                store_context,
                display_mode,
                cmd,
            );
        }
    }

    /// If we're on web and use web history this updates the
    /// web address bar and updates history.
    ///
    /// Otherwise this updates the viewer tracked history.
    fn update_history(&mut self, store_hub: &StoreHub) {
        if !self.startup_options().web_history_enabled() {
            self.update_viewer_history(store_hub);
        } else {
            // We don't want to spam the web history API with changes, because
            // otherwise it will start complaining about it being an insecure
            // operation.
            //
            // This is a kind of hacky way to fix that: If there are currently any
            // inputs, don't update the web address bar. This works for most cases
            // because you need to hold down pointer to aggressively scrub, need to
            // hold down key inputs to quickly step through the timeline.
            #[cfg(target_arch = "wasm32")]
            if !self.egui_ctx.is_using_pointer()
                && self
                    .egui_ctx
                    .input(|input| !input.any_touches() && input.keys_down.is_empty())
            {
                self.update_web_history(store_hub);
            }
        }
    }

    /// Updates the viewer tracked history
    fn update_viewer_history(&mut self, store_hub: &StoreHub) {
        let time_ctrl = store_hub
            .active_recording()
            .and_then(|db| self.state.time_control(db.store_id()));

        let display_mode = self.state.navigation.current();
        let selection = self.state.selection_state.selected_items();

        let Ok(url) =
            ViewerOpenUrl::from_context_expanded(store_hub, display_mode, time_ctrl, selection)
        else {
            return;
        };

        self.state.history.update_current_url(url);
    }

    /// Updates the web address and web history.
    #[cfg(target_arch = "wasm32")]
    fn update_web_history(&self, store_hub: &StoreHub) {
        let time_ctrl = store_hub
            .active_recording()
            .and_then(|db| self.state.time_control(db.store_id()));

        let display_mode = self.state.navigation.current();
        let selection = self.state.selection_state.selected_items();

        let Ok(url) =
            ViewerOpenUrl::from_context_expanded(store_hub, display_mode, time_ctrl, selection)
                .map(|mut url| {
                    // We don't want to update the url while playing, so we use the last paused time.
                    if let Some(fragment) = url.fragment_mut() {
                        fragment.when = time_ctrl.and_then(|time_ctrl| {
                            Some((
                                *time_ctrl.timeline_name(),
                                re_log_types::TimeCell {
                                    typ: time_ctrl.time_type()?,
                                    value: time_ctrl.last_paused_time()?.floor().into(),
                                },
                            ))
                        });
                    }

                    url
                })
                // History entries expect the url parameter, not the full url, therefore don't pass a base url.
                .and_then(|url| url.sharable_url(None))
        else {
            return;
        };

        re_log::debug!("Updating navigation bar");

        use crate::web_history::{HistoryEntry, HistoryExt as _, history};
        use crate::web_tools::JsResultExt as _;

        /// Returns the url without the fragment
        fn strip_fragment(url: &str) -> &str {
            // Split by url code for '#', which is used for fragments.
            url.rsplit_once("%23").map_or(url, |(url, _)| url)
        }

        if let Some(history) = history().ok_or_log_js_error() {
            let current_entry = history.current_entry().ok_or_log_js_error().flatten();
            let new_entry = HistoryEntry::new(url);
            if Some(&new_entry) != current_entry.as_ref() {
                // If only the fragment has changed, we replace history instead of pushing it.
                if current_entry
                    .and_then(|entry| {
                        Some((
                            entry.to_query_string().ok_or_log_js_error()?,
                            new_entry.to_query_string().ok_or_log_js_error()?,
                        ))
                    })
                    .is_some_and(|(current, new)| strip_fragment(&current) == strip_fragment(&new))
                {
                    history.replace_entry(new_entry).ok_or_log_js_error();
                } else {
                    history.push_entry(new_entry).ok_or_log_js_error();
                }
            }
        }
    }

    fn run_system_command(
        &mut self,
        sent_from: &std::panic::Location<'_>, // Who sent this command? Useful for debugging!
        cmd: SystemCommand,
        store_hub: &mut StoreHub,
        egui_ctx: &egui::Context,
    ) {
        match cmd {
            SystemCommand::TimeControlCommands {
                store_id,
                time_commands,
            } => {
                match store_id.kind() {
                    StoreKind::Recording => {
                        store_hub.set_active_recording_id(store_id.clone()); // Switch to this recording
                        let (storage_ctx, store_ctx) = store_hub.read_context(); // Materialize the target blueprint on-demand
                        if let Some(store_ctx) = store_ctx {
                            let target_blueprint = store_ctx.blueprint;
                            let blueprint_query =
                                self.state.blueprint_query_for_viewer(target_blueprint);

                            let blueprint_ctx = AppBlueprintCtx {
                                command_sender: &self.command_sender,
                                current_blueprint: target_blueprint,
                                default_blueprint: storage_ctx
                                    .hub
                                    .default_blueprint_for_app(store_id.application_id()),
                                blueprint_query,
                            };

                            let time_ctrl = self
                                .state
                                .time_control_mut(store_ctx.recording, &blueprint_ctx);

                            let response = time_ctrl.handle_time_commands(
                                Some(&blueprint_ctx),
                                store_ctx.recording.timeline_histograms(),
                                &time_commands,
                            );

                            if response.needs_repaint == NeedsRepaint::Yes {
                                self.egui_ctx.request_repaint();
                            }

                            handle_time_ctrl_event(
                                store_ctx.recording,
                                self.event_dispatcher.as_ref(),
                                &response,
                            );
                        } else {
                            re_log::error!(
                                "Skipping time control command because of missing blueprint"
                            );
                        }
                    }
                    StoreKind::Blueprint => {
                        if let Some(target_store) = store_hub.store_bundle().get(&store_id) {
                            let blueprint_ctx: Option<&AppBlueprintCtx<'_>> = None;
                            let response = self.state.blueprint_time_control.handle_time_commands(
                                blueprint_ctx,
                                target_store.timeline_histograms(),
                                &time_commands,
                            );

                            if response.needs_repaint == NeedsRepaint::Yes {
                                self.egui_ctx.request_repaint();
                            }
                        }
                    }
                }
            }
            SystemCommand::SetUrlFragment { store_id, fragment } => {
                // This adds new system commands, which will be handled later in the loop.
                self.go_to_dataset_data(store_id, fragment);
            }
            SystemCommand::CopyViewerUrl(url) => {
                if cfg!(target_arch = "wasm32") {
                    match combine_with_base_url(
                        self.startup_options.web_viewer_base_url().as_ref(),
                        [url],
                    ) {
                        Ok(url) => {
                            self.copy_text(url);
                        }
                        Err(err) => {
                            re_log::error!("{err}");
                        }
                    }
                } else {
                    self.copy_text(url);
                }
            }
            SystemCommand::ActivateApp(app_id) => {
                store_hub.set_active_app(app_id);
                if let Some(recording_id) = store_hub.active_store_id() {
                    self.state
                        .navigation
                        .replace(DisplayMode::LocalRecordings(recording_id.clone()));
                } else {
                    self.state.navigation.reset();
                }
            }

            SystemCommand::CloseApp(app_id) => {
                store_hub.close_app(&app_id);
            }

            SystemCommand::ActivateRecordingOrTable(entry) => {
                match &entry {
                    RecordingOrTable::Recording { store_id } => {
                        store_hub.set_active_recording_id(store_id.clone());
                    }
                    RecordingOrTable::Table { .. } => {}
                }
                self.state.navigation.replace(entry.display_mode());
            }

            SystemCommand::CloseRecordingOrTable(entry) => {
                // TODO(#9464): Find a better successor here.

                let data_source = match &entry {
                    RecordingOrTable::Recording { store_id } => {
                        store_hub.entity_db_mut(store_id).data_source.clone()
                    }
                    RecordingOrTable::Table { .. } => None,
                };
                if let Some(data_source) = data_source {
                    // Only certain sources should be closed.
                    #[expect(clippy::match_same_arms)]
                    let should_close = match &data_source {
                        // Specific files should stop streaming when closing them.
                        LogSource::File(_) => true,

                        // Specific HTTP streams should stop streaming when closing them.
                        LogSource::RrdHttpStream { .. } => true,

                        // Specific GRPC streams should stop streaming when closing them.
                        // TODO(#10967): We still stream in some data after that.
                        LogSource::RedapGrpcStream { .. } => true,

                        // Don't close generic connections (like to an SDK) that may feed in different recordings over time.
                        LogSource::RrdWebEvent
                        | LogSource::JsChannel { .. }
                        | LogSource::Sdk
                        | LogSource::Stdin
                        | LogSource::MessageProxy(_) => false,
                    };

                    if should_close {
                        self.rx_log.retain(|r| r.source() != &data_source);
                    }
                }

                store_hub.remove(&entry);
            }

            SystemCommand::CloseAllEntries => {
                self.state.navigation.reset();
                store_hub.clear_entries();

                // Stop receiving into the old recordings.
                // This is most important when going back to the example screen by using the "Back"
                // button in the browser, and there is still a connection downloading an .rrd.
                // That's the case of `LogSource::RrdHttpStream`.
                // TODO(emilk): exactly what things get kept and what gets cleared?
                self.rx_log.retain(|r| match r.source() {
                    LogSource::File(_) | LogSource::RrdHttpStream { .. } => false,

                    LogSource::JsChannel { .. }
                    | LogSource::RrdWebEvent
                    | LogSource::Sdk
                    | LogSource::RedapGrpcStream { .. }
                    | LogSource::MessageProxy { .. }
                    | LogSource::Stdin => true,
                });
            }

            SystemCommand::AddReceiver(rx) => {
                re_log::debug!("Received AddReceiver");
                self.add_log_receiver(rx);
            }

            SystemCommand::ChangeDisplayMode(display_mode) => {
                if &display_mode == self.state.navigation.current() {
                    return;
                }

                // Suppress loading screen if we're loading a recording that's already loaded, even if only partially.
                if let DisplayMode::Loading(source) = &display_mode
                    && let Some(re_uri::RedapUri::DatasetData(dataset_uri)) = source.redap_uri()
                    && store_hub
                        .store_bundle()
                        .entity_dbs()
                        .any(|db| db.store_id() == &dataset_uri.store_id())
                {
                    return;
                }

                if matches!(display_mode, DisplayMode::Loading(_)) {
                    self.state
                        .selection_state
                        .set_selection(re_viewer_context::ItemCollection::default());
                }
                self.state.navigation.replace(display_mode);

                egui_ctx.request_repaint(); // Make sure we actually see the new mode.
            }

            SystemCommand::OpenSettings => {
                self.state
                    .navigation
                    .replace(DisplayMode::Settings(Box::new(
                        self.state.navigation.current().clone(),
                    )));

                #[cfg(feature = "analytics")]
                re_analytics::record(|| re_analytics::event::SettingsOpened {});
            }

            SystemCommand::OpenChunkStoreBrowser => match self.state.navigation.current() {
                DisplayMode::LocalRecordings(_)
                | DisplayMode::RedapEntry(_)
                | DisplayMode::RedapServer(_) => {
                    self.state
                        .navigation
                        .replace(DisplayMode::ChunkStoreBrowser(Box::new(
                            self.state.navigation.current().clone(),
                        )));
                }

                DisplayMode::ChunkStoreBrowser(_)
                | DisplayMode::Settings(_)
                | DisplayMode::Loading(_)
                | DisplayMode::LocalTable(_) => {
                    re_log::debug!(
                        "Cannot activate chunk store browser from current display mode: {:?}",
                        self.state.navigation.current()
                    );
                }
            },

            SystemCommand::ResetDisplayMode => {
                self.state.navigation.reset();

                egui_ctx.request_repaint(); // Make sure we actually see the new mode.
            }

            SystemCommand::AddRedapServer(origin) => {
                if origin == *re_redap_browser::EXAMPLES_ORIGIN {
                    return;
                }
                if self.state.redap_servers.has_server(&origin) {
                    return;
                }

                self.state.redap_servers.add_server(origin.clone());

                if self
                    .store_hub
                    .as_ref()
                    .is_none_or(|store_hub| store_hub.active_recording_or_table().is_none())
                {
                    self.state
                        .navigation
                        .replace(DisplayMode::RedapServer(origin));
                }
                self.command_sender.send_ui(UICommand::ExpandBlueprintPanel);
            }

            SystemCommand::EditRedapServerModal(command) => {
                self.state.redap_servers.open_edit_server_modal(command);
            }

            SystemCommand::LoadDataSource(data_source) => {
                self.load_data_source(store_hub, egui_ctx, &data_source);
            }

            SystemCommand::ResetViewer => self.reset_viewer(store_hub, egui_ctx),
            SystemCommand::ClearActiveBlueprintAndEnableHeuristics => {
                re_log::debug!("Clear and generate new blueprint");
                store_hub.clear_active_blueprint_and_generate();
                egui_ctx.request_repaint(); // Many changes take a frame delay to show up.
            }
            SystemCommand::ClearActiveBlueprint => {
                // By clearing the blueprint the default blueprint will be restored
                // at the beginning of the next frame.
                re_log::debug!("Reset blueprint to default");
                store_hub.clear_active_blueprint();
                egui_ctx.request_repaint(); // Many changes take a frame delay to show up.
            }

            SystemCommand::AppendToStore(store_id, chunks) => {
                re_log::trace!(
                    "{}:{} Update {} entities: {}",
                    sent_from.file(),
                    sent_from.line(),
                    store_id.kind(),
                    chunks.iter().map(|c| c.entity_path()).join(", ")
                );

                let db = store_hub.entity_db_mut(&store_id);

                // No need to clear undo buffer if we're just appending static data.
                //
                // It would be nice to be able to undo edits to a recording, but
                // we haven't implemented that yet.
                if store_id.is_blueprint() && chunks.iter().any(|c| !c.is_static()) {
                    self.state
                        .blueprint_undo_state
                        .entry(store_id.clone())
                        .or_default()
                        .clear_redo_buffer(db);

                    if self.app_options().inspect_blueprint_timeline {
                        self.command_sender
                            .send_system(SystemCommand::TimeControlCommands {
                                store_id,
                                time_commands: vec![TimeControlCommand::SetPlayState(
                                    PlayState::Following,
                                )],
                            });
                    }
                }

                for chunk in chunks {
                    match db.add_chunk(&Arc::new(chunk)) {
                        Ok(_store_events) => {}
                        Err(err) => {
                            re_log::warn_once!("Failed to append chunk: {err}");
                        }
                    }
                }
            }

            SystemCommand::UndoBlueprint { blueprint_id } => {
                let inspect_blueprint_timeline = self.app_options().inspect_blueprint_timeline;
                let blueprint_db = store_hub.entity_db_mut(&blueprint_id);
                let undo_state = self
                    .state
                    .blueprint_undo_state
                    .entry(blueprint_id.clone())
                    .or_default();

                undo_state.undo(blueprint_db);

                // Update blueprint inspector timeline.
                if inspect_blueprint_timeline {
                    if let Some(redo_time) = undo_state.redo_time() {
                        self.command_sender
                            .send_system(SystemCommand::TimeControlCommands {
                                store_id: blueprint_id,
                                time_commands: vec![
                                    TimeControlCommand::SetPlayState(PlayState::Paused),
                                    TimeControlCommand::SetTime(redo_time.into()),
                                ],
                            });
                    } else {
                        self.command_sender
                            .send_system(SystemCommand::TimeControlCommands {
                                store_id: blueprint_id,
                                time_commands: vec![TimeControlCommand::SetPlayState(
                                    PlayState::Following,
                                )],
                            });
                    }
                }
            }
            SystemCommand::RedoBlueprint { blueprint_id } => {
                let inspect_blueprint_timeline = self.app_options().inspect_blueprint_timeline;
                let undo_state = self
                    .state
                    .blueprint_undo_state
                    .entry(blueprint_id.clone())
                    .or_default();

                undo_state.redo();

                // Update blueprint inspector timeline.
                if inspect_blueprint_timeline {
                    if let Some(redo_time) = undo_state.redo_time() {
                        self.command_sender
                            .send_system(SystemCommand::TimeControlCommands {
                                store_id: blueprint_id,
                                time_commands: vec![
                                    TimeControlCommand::SetPlayState(PlayState::Paused),
                                    TimeControlCommand::SetTime(redo_time.into()),
                                ],
                            });
                    } else {
                        self.command_sender
                            .send_system(SystemCommand::TimeControlCommands {
                                store_id: blueprint_id,
                                time_commands: vec![TimeControlCommand::SetPlayState(
                                    PlayState::Following,
                                )],
                            });
                    }
                }
            }

            SystemCommand::DropEntity(blueprint_id, entity_path) => {
                let blueprint_db = store_hub.entity_db_mut(&blueprint_id);
                blueprint_db.drop_entity_path_recursive(&entity_path);
            }

            #[cfg(debug_assertions)]
            SystemCommand::EnableInspectBlueprintTimeline(show) => {
                self.app_options_mut().inspect_blueprint_timeline = show;
            }

            SystemCommand::SetSelection(set) => {
                if let Some(item) = set.selection.single_item() {
                    // If the selected item has its own page, switch to it.
                    if let Some(display_mode) = DisplayMode::from_item(item) {
                        if let DisplayMode::LocalRecordings(store_id) = &display_mode {
                            store_hub.set_active_recording_id(store_id.clone());
                        }
                        self.state.navigation.replace(display_mode);
                    }
                }

                self.state.selection_state.set_selection(set);
                egui_ctx.request_repaint(); // Make sure we actually see the new selection.
            }

            SystemCommand::SetFocus(item) => {
                self.state.focused_item = Some(item);
            }

            SystemCommand::ShowNotification(notification) => {
                self.notifications.add(notification);
            }

            #[cfg(not(target_arch = "wasm32"))]
            SystemCommand::FileSaver(file_saver) => {
                if let Err(err) = self.background_tasks.spawn_file_saver(file_saver) {
                    re_log::error!("Failed to save file: {err}");
                }
            }

            SystemCommand::OnAuthChanged(auth) => {
                self.state.auth_state = auth;
            }

            SystemCommand::SetAuthCredentials {
                access_token,
                email,
            } => {
                let credentials =
                    match re_auth::oauth::Credentials::try_new(access_token, None, email) {
                        Ok(credentials) => credentials,
                        Err(err) => {
                            re_log::error!("Failed to create credentials: {err}");
                            return;
                        }
                    };
                if let Err(err) = credentials.ensure_stored() {
                    re_log::error!("Failed to store credentials: {err}");
                }
            }
            SystemCommand::Logout => {
                if let Err(err) = re_auth::oauth::clear_credentials() {
                    re_log::error!("Failed to logout: {err}");
                }
                self.state.redap_servers.logout();
            }
        }
    }

    pub fn auth_error_handler(sender: CommandSender) -> AuthErrorHandler {
        Arc::new(move |url, _err| {
            sender.send_system(SystemCommand::EditRedapServerModal(
                EditRedapServerModalCommand {
                    origin: url.origin.clone(),
                    open_on_success: Some(url.to_string()),
                    title: Some("Authenticate to see this recording".to_owned()),
                },
            ));
        })
    }

    /// Loads a data source into the viewer.
    ///
    /// Tries to detect whether the datasource is already present (either still streaming in or already loaded),
    /// and if so, will not load the data again.
    /// Instead, it will only perform any kind of selection/mode-switching operations associated with loading the given data source.
    ///
    /// Note that we *do not* change the display mode here _unconditionally_.
    /// For instance if the datasource is a blueprint for a dataset that may be loaded later,
    /// we don't want to switch out to it while the user browses a server.
    fn load_data_source(
        &mut self,
        store_hub: &mut StoreHub,
        egui_ctx: &egui::Context,
        data_source: &LogDataSource,
    ) {
        // Check if we've already loaded this data source and should just switch to it.
        //
        // Go through all sources that are still loading and those that are already in the store_hub.
        // (if we look only at the one from the store_hub, we might miss those that haven't hit it yet)
        let active_sources = self.rx_log.sources();
        let store_sources = store_hub
            .store_bundle()
            .entity_dbs()
            .filter_map(|db| db.data_source.as_ref());
        let mut all_sources = store_sources.chain(active_sources.iter().map(|s| s.as_ref()));

        match data_source {
            LogDataSource::RrdHttpUrl { url, follow } => {
                let new_source = LogSource::RrdHttpStream {
                    url: url.to_string(),
                    follow: *follow,
                };

                if all_sources.any(|source| source.is_same_ignoring_uri_fragments(&new_source)) {
                    if let Some(entity_db) = store_hub.find_recording_store_by_source(&new_source) {
                        if *follow {
                            self.command_sender
                                .send_system(SystemCommand::TimeControlCommands {
                                    store_id: entity_db.store_id().clone(),
                                    time_commands: vec![TimeControlCommand::SetPlayState(
                                        PlayState::Following,
                                    )],
                                });
                        }

                        let store_id = entity_db.store_id().clone();
                        debug_assert!(store_id.is_recording()); // `find_recording_store_by_source` should have filtered for recordings rather than blueprints.
                        drop(all_sources);
                        self.make_store_active_and_highlight(store_hub, egui_ctx, &store_id);
                    }
                    return;
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            LogDataSource::FilePath(_file_source, path) => {
                let new_source = LogSource::File(path.clone());
                if all_sources.any(|source| source.is_same_ignoring_uri_fragments(&new_source)) {
                    drop(all_sources);
                    self.try_make_recording_from_source_active(egui_ctx, store_hub, &new_source);
                    return;
                }
            }

            LogDataSource::FileContents(_file_source, _file_contents) => {
                // For raw file contents we currently can't determine whether we're already receiving them.
            }

            #[cfg(not(target_arch = "wasm32"))]
            LogDataSource::Stdin => {
                let new_source = LogSource::Stdin;
                if all_sources.any(|source| source.is_same_ignoring_uri_fragments(&new_source)) {
                    drop(all_sources);
                    self.try_make_recording_from_source_active(egui_ctx, store_hub, &new_source);
                    return;
                }
            }

            LogDataSource::RedapDatasetSegment {
                uri,
                select_when_loaded,
            } => {
                let new_source = LogSource::RedapGrpcStream {
                    uri: uri.clone(),
                    select_when_loaded: *select_when_loaded,
                };
                if all_sources.any(|source| source.is_same_ignoring_uri_fragments(&new_source)) {
                    // We're already receiving from the exact same data source!
                    // But we still should select if requested according to the fragments if any.
                    if *select_when_loaded {
                        // First make the recording itself active.
                        // `go_to_dataset_data` may override the selection again, but this is important regardless,
                        // since `go_to_dataset_data` does not change the active recording.
                        drop(all_sources);
                        self.make_store_active_and_highlight(store_hub, egui_ctx, &uri.store_id());
                    }

                    // Note that applying the fragment changes the per-recording settings like the active time cursor.
                    // Therefore, we apply it even when `select_when_loaded` is false.
                    self.go_to_dataset_data(uri.store_id(), uri.fragment.clone());

                    return;
                }
            }

            LogDataSource::RedapProxy(uri) => {
                let new_source = LogSource::MessageProxy(uri.clone());
                if all_sources.any(|source| source.is_same_ignoring_uri_fragments(&new_source)) {
                    drop(all_sources);
                    self.try_make_recording_from_source_active(egui_ctx, store_hub, &new_source);
                    return;
                }
            }
        }

        let sender = self.command_sender.clone();
        let stream = data_source
            .clone()
            .stream(Self::auth_error_handler(sender), &self.connection_registry);

        #[cfg(feature = "analytics")]
        if let Some(analytics) = re_analytics::Analytics::global_or_init() {
            let data_source_analytics = data_source.analytics();
            analytics.record(re_analytics::event::LoadDataSource {
                source_type: data_source_analytics.source_type,
                file_extension: data_source_analytics.file_extension,
                file_source: data_source_analytics.file_source,
                started_successfully: stream.is_ok(),
            });
        }

        match stream {
            Ok(rx) => self.add_log_receiver(rx),
            Err(err) => {
                re_log::error!("Failed to open data source: {}", re_error::format(err));
            }
        }
    }

    /// Applies a fragment.
    ///
    /// Does *not* switch the active recording.
    fn go_to_dataset_data(&self, store_id: StoreId, fragment: re_uri::Fragment) {
        let re_uri::Fragment {
            selection,
            when,
            time_selection,
        } = fragment;

        if let Some(selection) = selection {
            let re_log_types::DataPath {
                entity_path,
                instance,
                component,
            } = selection;

            let item = if let Some(component) = component {
                Item::from(re_log_types::ComponentPath::new(entity_path, component))
            } else if let Some(instance) = instance {
                Item::from(InstancePath::instance(entity_path, instance))
            } else {
                Item::from(entity_path)
            };

            self.command_sender
                .send_system(SystemCommand::set_selection(item.clone()));
        }

        let mut time_commands = Vec::new();
        if let Some(time_selection) = time_selection {
            time_commands.push(TimeControlCommand::SetActiveTimeline(
                *time_selection.timeline.name(),
            ));
            time_commands.push(TimeControlCommand::SetTimeSelection(time_selection.range));
            time_commands.push(TimeControlCommand::SetLoopMode(LoopMode::Selection));
        }

        if let Some((timeline, timecell)) = when {
            time_commands.push(TimeControlCommand::SetActiveTimeline(timeline));
            time_commands.push(TimeControlCommand::SetPlayState(PlayState::Paused));
            time_commands.push(TimeControlCommand::SetTime(timecell.value.into()));
        }

        if !time_commands.is_empty() {
            self.command_sender
                .send_system(SystemCommand::TimeControlCommands {
                    store_id,
                    time_commands,
                });
        }
    }

    fn run_ui_command(
        &mut self,
        egui_ctx: &egui::Context,
        app_blueprint: &AppBlueprint<'_>,
        storage_context: &StorageContext<'_>,
        store_context: Option<&StoreContext<'_>>,
        display_mode: &DisplayMode,
        cmd: UICommand,
    ) {
        let mut force_store_info = false;
        let active_store_id = store_context
            .map(|ctx| ctx.recording_store_id().clone())
            // Don't redirect data to the welcome screen.
            .filter(|store_id| store_id.application_id() != &StoreHub::welcome_screen_app_id())
            .unwrap_or_else(|| {
                // If we don't have any application ID to recommend (which means we are on the welcome screen),
                // then just generate a new one using a UUID.
                let application_id = ApplicationId::random();

                // NOTE: We don't override blueprints' store IDs anyhow, so it is sound to assume that
                // this can only be a recording.
                let recording_id = RecordingId::random();

                // We're creating a recording just-in-time, directly from the viewer.
                // We need those store infos or the data will just be silently ignored.
                force_store_info = true;

                StoreId::recording(application_id, recording_id)
            });

        match cmd {
            UICommand::SaveRecording => {
                #[cfg(target_arch = "wasm32")] // Web
                {
                    if let Err(err) = save_active_recording(self, store_context, None) {
                        re_log::error!("Failed to save recording: {err}");
                    }
                }

                #[cfg(not(target_arch = "wasm32"))] // Native
                {
                    let mut selected_stores = vec![];
                    for item in self.state.selection_state.selected_items().iter_items() {
                        match item {
                            Item::AppId(selected_app_id) => {
                                for recording in storage_context.bundle.recordings() {
                                    if recording.application_id() == selected_app_id {
                                        selected_stores.push(recording.store_id().clone());
                                    }
                                }
                            }
                            Item::StoreId(store_id) => {
                                selected_stores.push(store_id.clone());
                            }
                            _ => {}
                        }
                    }

                    let selected_stores = selected_stores
                        .iter()
                        .filter_map(|store_id| storage_context.bundle.get(store_id))
                        .collect_vec();

                    if selected_stores.is_empty() {
                        if let Err(err) = save_active_recording(self, store_context, None) {
                            re_log::error!("Failed to save recording: {err}");
                        }
                    } else if selected_stores.len() == 1 {
                        // Common case: saving a single recording.
                        // In this case we want the user to be able to pick a file name (not just a folder):
                        if let Err(err) = save_recording(self, selected_stores[0], None) {
                            re_log::error!("Failed to save recording: {err}");
                        }
                    } else {
                        // Save all selected recordings to a folder:
                        if let Some(folder) = rfd::FileDialog::new()
                            .set_title("Save recordings to folder")
                            .pick_folder()
                        {
                            self.save_many_recordings(&selected_stores, &folder);
                        } else {
                            re_log::info!("No folder selected - recordings not saved.");
                        }
                    }
                }
            }
            UICommand::SaveRecordingSelection => {
                if let Err(err) = save_active_recording(
                    self,
                    store_context,
                    self.state.loop_selection(store_context),
                ) {
                    re_log::error!("Failed to save recording: {err}");
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::SaveVideoSelection => {
                if let Err(err) = save_cropped_video(self, store_context) {
                    re_log::error!("Failed to save cropped video: {err}");
                }
            }
            #[cfg(target_arch = "wasm32")]
            UICommand::SaveVideoSelection => {
                re_log::warn!("Saving cropped videos is not supported in the web viewer.");
            }

            UICommand::SaveBlueprint => {
                if let Err(err) = save_blueprint(self, store_context) {
                    re_log::error!("Failed to save blueprint: {err}");
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::Open => {
                for file_path in open_file_dialog_native(self.main_thread_token) {
                    self.command_sender
                        .send_system(SystemCommand::LoadDataSource(LogDataSource::FilePath(
                            FileSource::FileDialog {
                                recommended_store_id: None,
                                force_store_info,
                            },
                            file_path,
                        )));
                }
            }
            #[cfg(target_arch = "wasm32")]
            UICommand::Open => {
                let egui_ctx = egui_ctx.clone();

                let promise = poll_promise::Promise::spawn_local(async move {
                    let file = async_open_rrd_dialog().await;
                    egui_ctx.request_repaint(); // Wake ui thread
                    file
                });

                self.open_files_promise = Some(PendingFilePromise {
                    recommended_store_id: None,
                    force_store_info,
                    promise,
                });
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::Import => {
                for file_path in open_file_dialog_native(self.main_thread_token) {
                    self.command_sender
                        .send_system(SystemCommand::LoadDataSource(LogDataSource::FilePath(
                            FileSource::FileDialog {
                                recommended_store_id: Some(active_store_id.clone()),
                                force_store_info,
                            },
                            file_path,
                        )));
                }
            }
            #[cfg(target_arch = "wasm32")]
            UICommand::Import => {
                let egui_ctx = egui_ctx.clone();

                let promise = poll_promise::Promise::spawn_local(async move {
                    let file = async_open_rrd_dialog().await;
                    egui_ctx.request_repaint(); // Wake ui thread
                    file
                });

                self.open_files_promise = Some(PendingFilePromise {
                    recommended_store_id: Some(active_store_id.clone()),
                    force_store_info,
                    promise,
                });
            }

            UICommand::OpenUrl => {
                self.state.open_url_modal.open();
            }

            UICommand::CloseCurrentRecording => {
                let cur_rec = store_context.map(|ctx| ctx.recording.store_id());
                if let Some(cur_rec) = cur_rec {
                    self.command_sender
                        .send_system(SystemCommand::CloseRecordingOrTable(cur_rec.clone().into()));
                }
            }
            UICommand::CloseAllEntries => {
                self.command_sender
                    .send_system(SystemCommand::CloseAllEntries);
            }

            UICommand::NextRecording => {
                self.state
                    .recording_panel
                    .send_command(re_recording_panel::RecordingPanelCommand::SelectNextRecording);
            }
            UICommand::PreviousRecording => {
                self.state.recording_panel.send_command(
                    re_recording_panel::RecordingPanelCommand::SelectPreviousRecording,
                );
            }

            UICommand::NavigateBack => {
                if let Some(url) = self.state.history.go_back() {
                    url.clone().open(
                        egui_ctx,
                        &OpenUrlOptions {
                            follow_if_http: true,
                            select_redap_source_when_loaded: true,
                            show_loader: true,
                        },
                        &self.command_sender,
                    );
                }
            }
            UICommand::NavigateForward => {
                if let Some(url) = self.state.history.go_forward() {
                    url.clone().open(
                        egui_ctx,
                        &OpenUrlOptions {
                            follow_if_http: true,
                            select_redap_source_when_loaded: true,
                            show_loader: true,
                        },
                        &self.command_sender,
                    );
                }
            }

            UICommand::Undo => {
                if let Some(store_context) = store_context {
                    let blueprint_id = store_context.blueprint.store_id().clone();
                    self.command_sender
                        .send_system(SystemCommand::UndoBlueprint { blueprint_id });
                }
            }
            UICommand::Redo => {
                if let Some(store_context) = store_context {
                    let blueprint_id = store_context.blueprint.store_id().clone();
                    self.command_sender
                        .send_system(SystemCommand::RedoBlueprint { blueprint_id });
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::Quit => {
                egui_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }

            UICommand::OpenWebHelp => {
                egui_ctx.open_url(egui::output::OpenUrl {
                    url: "https://www.rerun.io/docs/getting-started/navigating-the-viewer"
                        .to_owned(),
                    new_tab: true,
                });
            }

            UICommand::OpenRerunDiscord => {
                egui_ctx.open_url(egui::output::OpenUrl {
                    url: "https://discord.gg/PXtCgFBSmH".to_owned(),
                    new_tab: true,
                });
            }

            UICommand::ResetViewer => self.command_sender.send_system(SystemCommand::ResetViewer),
            UICommand::ClearActiveBlueprint => {
                self.command_sender
                    .send_system(SystemCommand::ClearActiveBlueprint);
            }
            UICommand::ClearActiveBlueprintAndEnableHeuristics => {
                self.command_sender
                    .send_system(SystemCommand::ClearActiveBlueprintAndEnableHeuristics);
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::OpenProfiler => {
                self.profiler.start();
            }

            UICommand::ToggleMemoryPanel => {
                self.memory_panel_open ^= true;
            }
            UICommand::TogglePanelStateOverrides => {
                self.panel_state_overrides_active ^= true;
            }
            UICommand::ToggleTopPanel => {
                app_blueprint.toggle_top_panel(&self.command_sender);
            }
            UICommand::ToggleBlueprintPanel => {
                app_blueprint.toggle_blueprint_panel(&self.command_sender);
            }
            UICommand::ExpandBlueprintPanel => {
                if !app_blueprint.blueprint_panel_state().is_expanded() {
                    app_blueprint.toggle_blueprint_panel(&self.command_sender);
                }
            }
            UICommand::ToggleSelectionPanel => {
                app_blueprint.toggle_selection_panel(&self.command_sender);
            }
            UICommand::ExpandSelectionPanel => {
                if !app_blueprint.selection_panel_state().is_expanded() {
                    app_blueprint.toggle_selection_panel(&self.command_sender);
                }
            }
            UICommand::ToggleTimePanel => app_blueprint.toggle_time_panel(&self.command_sender),

            UICommand::ToggleChunkStoreBrowser => match self.state.navigation.current() {
                DisplayMode::LocalRecordings(_)
                | DisplayMode::RedapEntry(_)
                | DisplayMode::RedapServer(_) => {
                    self.state
                        .navigation
                        .replace(DisplayMode::ChunkStoreBrowser(Box::new(
                            self.state.navigation.current().clone(),
                        )));
                }

                DisplayMode::ChunkStoreBrowser(mode) => {
                    self.state.navigation.replace((**mode).clone());
                }

                DisplayMode::Settings(_) | DisplayMode::Loading(_) | DisplayMode::LocalTable(_) => {
                    re_log::debug!(
                        "Cannot toggle chunk store browser from current display mode: {:?}",
                        self.state.navigation.current()
                    );
                }
            },

            #[cfg(debug_assertions)]
            UICommand::ToggleBlueprintInspectionPanel => {
                self.app_options_mut().inspect_blueprint_timeline ^= true;
            }

            #[cfg(debug_assertions)]
            UICommand::ToggleEguiDebugPanel => {
                self.egui_debug_panel_open ^= true;
            }

            UICommand::ToggleFullscreen => {
                self.toggle_fullscreen();
            }

            UICommand::Settings => {
                self.command_sender.send_system(SystemCommand::OpenSettings);
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::ZoomIn => {
                let mut zoom_factor = egui_ctx.zoom_factor();
                zoom_factor += 0.1;
                zoom_factor = zoom_factor.clamp(MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR);
                zoom_factor = (zoom_factor * 10.).round() / 10.;
                egui_ctx.set_zoom_factor(zoom_factor);
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::ZoomOut => {
                let mut zoom_factor = egui_ctx.zoom_factor();
                zoom_factor -= 0.1;
                zoom_factor = zoom_factor.clamp(MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR);
                zoom_factor = (zoom_factor * 10.).round() / 10.;
                egui_ctx.set_zoom_factor(zoom_factor);
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::ZoomReset => {
                egui_ctx.set_zoom_factor(1.0);
            }

            UICommand::ToggleCommandPalette => {
                self.cmd_palette.toggle();
            }

            UICommand::PlaybackTogglePlayPause => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::TogglePlayPause],
                        });
                }
            }
            UICommand::PlaybackFollow => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::SetPlayState(
                                PlayState::Following,
                            )],
                        });
                }
            }
            UICommand::PlaybackStepBack => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::StepTimeBack],
                        });
                }
            }
            UICommand::PlaybackStepForward => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::StepTimeForward],
                        });
                }
            }
            UICommand::PlaybackBack => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveBySeconds(-0.1)],
                        });
                }
            }
            UICommand::PlaybackForward => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveBySeconds(0.1)],
                        });
                }
            }
            UICommand::PlaybackBackFast => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveBySeconds(-1.0)],
                        });
                }
            }
            UICommand::PlaybackForwardFast => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveBySeconds(1.0)],
                        });
                }
            }
            UICommand::PlaybackBeginning => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveBeginning],
                        });
                }
            }
            UICommand::PlaybackEnd => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::MoveEnd],
                        });
                }
            }
            UICommand::PlaybackRestart => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::Restart],
                        });
                }
            }

            UICommand::PlaybackSpeed(speed) => {
                if let Some(store_id) = storage_context.hub.active_store_id() {
                    self.command_sender
                        .send_system(SystemCommand::TimeControlCommands {
                            store_id: store_id.clone(),
                            time_commands: vec![TimeControlCommand::SetSpeed(speed.0.0)],
                        });
                }
            }

            #[cfg(not(target_arch = "wasm32"))]
            UICommand::ScreenshotWholeApp => {
                self.screenshotter.request_screenshot(egui_ctx);
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::PrintChunkStore => {
                if let Some(ctx) = store_context {
                    let text = format!("{}", ctx.recording.storage_engine().store());
                    egui_ctx.copy_text(text.clone());
                    println!("{text}");
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::PrintBlueprintStore => {
                if let Some(ctx) = store_context {
                    let text = format!("{}", ctx.blueprint.storage_engine().store());
                    egui_ctx.copy_text(text.clone());
                    println!("{text}");
                }
            }
            #[cfg(not(target_arch = "wasm32"))]
            UICommand::PrintPrimaryCache => {
                if let Some(ctx) = store_context {
                    let text = format!("{:?}", ctx.recording.storage_engine().cache());
                    egui_ctx.copy_text(text.clone());
                    println!("{text}");
                }
            }

            #[cfg(debug_assertions)]
            UICommand::ResetEguiMemory => {
                egui_ctx.memory_mut(|mem| *mem = Default::default());

                // re-apply style, which is lost when resetting memory
                re_ui::apply_style_and_install_loaders(egui_ctx);
            }

            UICommand::Share => {
                let selection = self.state.selection_state.selected_items();
                let rec_cfg = storage_context
                    .hub
                    .active_store_id()
                    .and_then(|id| self.state.time_controls.get(id));
                if let Err(err) = self.state.share_modal.open(
                    storage_context.hub,
                    display_mode,
                    rec_cfg,
                    selection,
                ) {
                    re_log::error!("Cannot share link to current screen: {err}");
                }
            }
            UICommand::CopyDirectLink => {
                match ViewerOpenUrl::from_display_mode(storage_context.hub, display_mode) {
                    Ok(url) => self.run_copy_link_command(&url),
                    Err(err) => re_log::error!("{err}"),
                }
            }

            UICommand::CopyTimeSelectionLink => {
                match ViewerOpenUrl::from_display_mode(storage_context.hub, display_mode) {
                    Ok(mut url) => {
                        if let Some(fragment) = url.fragment_mut() {
                            let time_ctrl = storage_context
                                .hub
                                .active_store_id()
                                .and_then(|id| self.state.time_control(id));

                            if let Some(time_ctrl) = &time_ctrl
                                && let Some(time_selection) = time_ctrl.time_selection()
                                && let Some(timeline) = time_ctrl.timeline()
                            {
                                fragment.time_selection = Some(re_uri::TimeSelection {
                                    timeline: *timeline,
                                    range: time_selection.to_int(),
                                });
                            } else {
                                re_log::warn!("No timeline selection to copy");
                            }
                        } else {
                            re_log::warn!(
                                "The current recording doesn't support sharing a time range"
                            );
                        }

                        self.run_copy_link_command(&url);
                    }
                    Err(err) => re_log::error!("{err}"),
                }
            }

            #[cfg(target_arch = "wasm32")]
            UICommand::RestartWithWebGl => {
                if crate::web_tools::set_url_parameter_and_refresh("renderer", "webgl").is_err() {
                    re_log::error!("Failed to set URL parameter `renderer=webgl` & refresh page.");
                }
            }

            #[cfg(target_arch = "wasm32")]
            UICommand::RestartWithWebGpu => {
                if crate::web_tools::set_url_parameter_and_refresh("renderer", "webgpu").is_err() {
                    re_log::error!("Failed to set URL parameter `renderer=webgpu` & refresh page.");
                }
            }

            UICommand::CopyEntityHierarchy => {
                self.copy_entity_hierarchy_to_clipboard(egui_ctx, store_context);
            }

            UICommand::AddRedapServer => {
                self.state.redap_servers.open_add_server_modal();
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_many_recordings(&mut self, stores: &[&EntityDb], folder: &std::path::Path) {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        use re_log::ResultExt as _;
        use tap::Pipe as _;

        re_tracing::profile_function!();

        let num_stores = stores.len();
        let any_error = Arc::new(AtomicBool::new(false));
        let num_remaining = Arc::new(AtomicUsize::new(stores.len()));

        re_log::info!("Saving {num_stores} recordings to {}…", folder.display());

        for store in stores {
            let messages = store.to_messages(None).collect_vec();

            let file_name = if let Some(rec_name) = store
                .recording_info_property::<re_sdk_types::components::Name>(
                    re_sdk_types::archetypes::RecordingInfo::descriptor_name().component,
                ) {
                rec_name.to_string()
            } else {
                format!("{}-{}", store.application_id(), store.recording_id())
            }
            .pipe(|name| sanitize_file_name(&name))
            .pipe(|stem| format!("{stem}.rrd"));

            let file_path = folder.join(file_name.clone());
            let any_error = any_error.clone();
            let num_remaining = num_remaining.clone();
            let folder = folder.display().to_string();

            self.background_tasks
                .spawn_threaded_promise(file_name, move || {
                    let res = crate::saving::encode_to_file(
                        re_build_info::CrateVersion::LOCAL,
                        &file_path,
                        messages.into_iter(),
                    );

                    if res.is_err() {
                        any_error.store(true, Ordering::Relaxed);
                    }

                    let num_remaining = num_remaining.fetch_sub(1, Ordering::Relaxed) - 1;

                    if num_remaining == 0 {
                        if any_error.load(Ordering::Relaxed) {
                            re_log::error!("Some recordings failed to save.");
                        } else {
                            re_log::info!("{num_stores} recordings successfully saved to {folder}");
                        }
                    }

                    res
                })
                .ok_or_log_error_once();
        }
    }

    fn run_copy_link_command(&mut self, content_url: &ViewerOpenUrl) {
        let base_url = self.startup_options.web_viewer_base_url();

        match content_url.sharable_url(base_url.as_ref()) {
            Ok(url) => {
                self.copy_text(url);
            }
            Err(err) => {
                re_log::error!("{err}");
            }
        }
    }

    /// Copies text to the clipboard, and gives a notification about it.
    fn copy_text(&mut self, url: String) {
        self.notifications
            .success(format!("Copied {url:?} to clipboard"));
        self.egui_ctx.copy_text(url);
    }

    fn copy_entity_hierarchy_to_clipboard(
        &mut self,
        egui_ctx: &egui::Context,
        store_context: Option<&StoreContext<'_>>,
    ) {
        let Some(entity_db) = store_context.as_ref().map(|ctx| ctx.recording) else {
            re_log::warn!("Could not copy entity hierarchy: No active recording");
            return;
        };

        let mut hierarchy_text = String::new();

        // Add application ID and recording ID header
        hierarchy_text.push_str(&format!(
            "Application ID: {}\nRecording ID: {}\n\n",
            entity_db.application_id(),
            entity_db.recording_id()
        ));

        hierarchy_text.push_str(&entity_db.format_with_components());

        if hierarchy_text.is_empty() {
            hierarchy_text = "(no entities)".to_owned();
        }

        egui_ctx.copy_text(hierarchy_text.clone());
        self.notifications
            .success("Copied entity hierarchy with schema to clipboard".to_owned());
    }

    fn memory_panel_ui(
        &self,
        ui: &mut egui::Ui,
        gpu_resource_stats: &WgpuResourcePoolStatistics,
        store_stats: Option<&StoreHubStats>,
    ) {
        let frame = egui::Frame {
            fill: ui.visuals().panel_fill,
            ..ui.tokens().bottom_panel_frame()
        };

        egui::TopBottomPanel::bottom("memory_panel")
            .default_height(300.0)
            .resizable(true)
            .frame(frame)
            .show_animated_inside(ui, self.memory_panel_open, |ui| {
                self.memory_panel.ui(
                    ui,
                    &self.startup_options.memory_limit,
                    gpu_resource_stats,
                    store_stats,
                );
            });
    }

    fn egui_debug_panel_ui(&self, ui: &mut egui::Ui) {
        let egui_ctx = ui.ctx().clone();

        egui::SidePanel::left("style_panel")
            .default_width(300.0)
            .resizable(true)
            .frame(ui.tokens().top_panel_frame())
            .show_animated_inside(ui, self.egui_debug_panel_open, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if ui
                        .button("request_discard")
                        .on_hover_text("Request a second layout pass. Just for testing.")
                        .clicked()
                    {
                        ui.ctx().request_discard("testing");
                    }

                    egui::CollapsingHeader::new("egui settings")
                        .default_open(false)
                        .show(ui, |ui| {
                            egui_ctx.settings_ui(ui);
                        });

                    egui::CollapsingHeader::new("egui inspection")
                        .default_open(false)
                        .show(ui, |ui| {
                            egui_ctx.inspection_ui(ui);
                        });
                });
            });
    }

    /// Top-level ui function.
    ///
    /// Shows the viewer ui.
    #[expect(clippy::too_many_arguments)]
    fn ui(
        &mut self,
        egui_ctx: &egui::Context,
        frame: &eframe::Frame,
        app_blueprint: &AppBlueprint<'_>,
        gpu_resource_stats: &WgpuResourcePoolStatistics,
        store_context: Option<&StoreContext<'_>>,
        storage_context: &StorageContext<'_>,
        store_stats: Option<&StoreHubStats>,
    ) {
        let mut main_panel_frame = egui::Frame::default();
        if re_ui::CUSTOM_WINDOW_DECORATIONS {
            // Add some margin so that we can later paint an outline around it all.
            main_panel_frame.inner_margin = 1.0.into();
        }

        egui::CentralPanel::default()
            .frame(main_panel_frame)
            .show(egui_ctx, |ui| {
                paint_background_fill(ui);

                crate::ui::mobile_warning_ui(ui);

                crate::ui::top_panel(
                    frame,
                    self,
                    app_blueprint,
                    store_context,
                    storage_context.hub,
                    gpu_resource_stats,
                    ui,
                );

                self.memory_panel_ui(ui, gpu_resource_stats, store_stats);

                self.egui_debug_panel_ui(ui);

                #[cfg(not(target_arch = "wasm32"))]
                if let Some(store_context) = store_context
                    && self.cropped_video_export.is_some()
                {
                    let frame_render_state = frame
                        .wgpu_render_state()
                        .expect("Failed to get frame render state");
                    self.run_offscreen_cropped_video_export(
                        frame_render_state,
                        store_context,
                        storage_context,
                    );
                }

                let egui_renderer = &mut frame
                    .wgpu_render_state()
                    .expect("Failed to get frame render state")
                    .renderer
                    .write();

                if let Some(render_ctx) = egui_renderer
                    .callback_resources
                    .get_mut::<re_renderer::RenderContext>()
                {
                    if let Some(store_context) = store_context {
                        render_ctx.begin_frame(); // This may actually be called multiple times per egui frame, if we have a multi-pass layout frame.

                        // In some (rare) circumstances we run two egui passes in a single frame.
                        // This happens on call to `egui::Context::request_discard`.
                        let is_start_of_new_frame = egui_ctx.current_pass_index() == 0;

                        if is_start_of_new_frame {
                            self.state.redap_servers.on_frame_start(
                                &self.connection_registry,
                                &self.async_runtime,
                                &self.egui_ctx,
                            );
                            self.collect_cropped_video_view_frames(render_ctx);
                        }

                        if is_start_of_new_frame {
                            if let Some((export_id, frame_index, view_ids)) =
                                self.prepare_cropped_video_export_frame_request()
                            {
                                self.renderer_video_export_state.set_request(
                                    export_id,
                                    frame_index,
                                    view_ids,
                                );
                            } else {
                                self.renderer_video_export_state.clear_request();
                            }
                        }

                        let mut startup_options = self.startup_options.clone();

                        self.state.show(
                            &self.app_env,
                            &mut startup_options,
                            app_blueprint,
                            ui,
                            render_ctx,
                            store_context,
                            storage_context,
                            &self.reflection,
                            &self.component_ui_registry,
                            &self.component_fallback_registry,
                            &self.view_class_registry,
                            &self.rx_log,
                            &self.command_sender,
                            &WelcomeScreenState {
                                hide_examples: self.startup_options.hide_welcome_screen,
                                opacity: self.welcome_screen_opacity(egui_ctx),
                            },
                            self.event_dispatcher.as_ref(),
                            &self.connection_registry,
                            &self.async_runtime,
                        );
                        self.startup_options = startup_options;
                        render_ctx.before_submit();
                    }

                    self.show_text_logs_as_notifications();
                }
            });

        if self.app_options().show_notification_toasts {
            self.notifications.show_toasts(egui_ctx);
        }
    }

    /// Show recent text log messages to the user as toast notifications.
    fn show_text_logs_as_notifications(&mut self) {
        re_tracing::profile_function!();

        while let Ok(message) = self.text_log_rx.try_recv() {
            self.notifications.add_log(message);
        }
    }

    fn receive_messages(&mut self, store_hub: &mut StoreHub, egui_ctx: &egui::Context) {
        re_tracing::profile_function!();

        let start = web_time::Instant::now();

        while let Some((channel_source, msg)) = self.rx_log.try_recv() {
            re_log::trace!("Received a message from {channel_source:?}"); // Used by `test_ui_wakeup` test app!

            let msg = match msg.payload {
                re_log_channel::SmartMessagePayload::Msg(msg) => msg,

                re_log_channel::SmartMessagePayload::Flush { on_flush_done } => {
                    on_flush_done();
                    continue;
                }

                re_log_channel::SmartMessagePayload::Quit(err) => {
                    if let Some(err) = err {
                        re_log::warn!("Data source {} has left unexpectedly: {err}", msg.source);
                    } else {
                        re_log::debug!("Data source {} has finished", msg.source);
                    }
                    continue;
                }
            };

            match msg {
                DataSourceMessage::RrdManifest(store_id, rrd_manifest) => {
                    let entity_db = store_hub.entity_db_mut(&store_id);
                    entity_db.add_rrd_manifest_message(*rrd_manifest);
                }

                DataSourceMessage::LogMsg(msg) => {
                    self.receive_log_msg(&msg, store_hub, egui_ctx, &channel_source);
                }

                DataSourceMessage::TableMsg(table) => {
                    self.receive_table_msg(store_hub, egui_ctx, table);
                }

                DataSourceMessage::UiCommand(ui_command) => {
                    self.receive_data_source_ui_command(ui_command, &channel_source);
                }
            }

            if start.elapsed() > web_time::Duration::from_millis(10) {
                egui_ctx.request_repaint(); // make sure we keep receiving messages asap
                break; // don't block the main thread for too long
            }
        }

        // Run pending system commands in case any of the messages resulted in additional commands.
        // This avoid further frame delays on these commands.
        self.run_pending_system_commands(store_hub, egui_ctx);
    }

    fn receive_log_msg(
        &self,
        msg: &LogMsg,
        store_hub: &mut StoreHub,
        egui_ctx: &egui::Context,
        channel_source: &LogSource,
    ) {
        let store_id = msg.store_id();

        if store_hub.is_active_blueprint(store_id) {
            // TODO(#5514): handle loading of active blueprints.
            re_log::warn_once!(
                "Loading a blueprint {store_id:?} that is active. See https://github.com/rerun-io/rerun/issues/5514 for details."
            );
        }

        // Note that the `SetStoreInfo` message might be missing. It's not strictly necessary to add a new store.
        let msg_will_add_new_store = !store_hub.store_bundle().contains(store_id);

        let entity_db = store_hub.entity_db_mut(store_id);
        if entity_db.data_source.is_none() {
            entity_db.data_source = Some((*channel_source).clone());
        }

        let was_empty = entity_db.is_empty();
        let entity_db_add_result = entity_db.add(msg);

        // Downgrade to read-only, so we can access caches.
        let entity_db = store_hub
            .entity_db(store_id)
            .expect("Just queried it mutable and that was fine.");

        match entity_db_add_result {
            Ok(store_events) => {
                if let Some(caches) = store_hub.active_caches() {
                    caches.on_store_events(&store_events, entity_db);
                }

                self.validate_loaded_events(&store_events);
            }

            Err(err) => {
                re_log::error_once!("Failed to add incoming msg: {err}");
            }
        }

        if was_empty && !entity_db.is_empty() {
            // Hack: we cannot go to a specific timeline or entity until we know about it.
            // Now we _hopefully_ do.
            if let LogSource::RedapGrpcStream { uri, .. } = channel_source {
                self.go_to_dataset_data(uri.store_id(), uri.fragment.clone());
            }
        }

        #[expect(clippy::match_same_arms)]
        match &msg {
            LogMsg::SetStoreInfo(_) => {
                // Causes a new store typically. But that's handled below via `on_new_store`.
            }

            LogMsg::ArrowMsg(_, _) => {
                // Handled by `EntityDb::add`.
            }

            LogMsg::BlueprintActivationCommand(cmd) => match store_id.kind() {
                StoreKind::Recording => {
                    re_log::debug!(
                        "Unexpected `BlueprintActivationCommand` message for {store_id:?}"
                    );
                }
                StoreKind::Blueprint => {
                    if let Some(info) = entity_db.store_info() {
                        re_log::trace!(
                            "Activating blueprint that was loaded from {channel_source}"
                        );
                        let app_id = info.application_id().clone();
                        if cmd.make_default {
                            store_hub
                                .set_default_blueprint_for_app(store_id)
                                .unwrap_or_else(|err| {
                                    re_log::warn!("Failed to make blueprint default: {err}");
                                });
                        }
                        if cmd.make_active {
                            store_hub
                                .set_cloned_blueprint_active_for_app(store_id)
                                .unwrap_or_else(|err| {
                                    re_log::warn!("Failed to make blueprint active: {err}");
                                });

                            // Switch to this app, e.g. on drag-and-drop of a blueprint file
                            store_hub.set_active_app(app_id);

                            // If the viewer is in the background, tell the user that it has received something new.
                            egui_ctx.send_viewport_cmd(
                                egui::ViewportCommand::RequestUserAttention(
                                    egui::UserAttentionType::Informational,
                                ),
                            );
                        }
                    } else {
                        re_log::warn!(
                            "Got ActivateStore message without first receiving a SetStoreInfo"
                        );
                    }
                }
            },
        }

        // Handle any action that is triggered by a new store _after_ processing the message that caused it.
        if msg_will_add_new_store {
            self.on_new_store(egui_ctx, store_id, channel_source, store_hub);
        }
    }

    fn receive_table_msg(
        &self,
        store_hub: &mut StoreHub,
        egui_ctx: &egui::Context,
        table: TableMsg,
    ) {
        let TableMsg { id, data } = table;

        // TODO(grtlr): For now we don't append anything to existing stores and always replace.
        // TODO(ab): When we actually append to existing table, we will have to clear the UI
        // cache by calling `DataFusionTableWidget::clear_state`.
        let store = TableStore::default();
        if let Err(err) = store.add_record_batch(data) {
            re_log::error!("Failed to load table {id}: {err}");
        } else {
            if store_hub.insert_table_store(id.clone(), store).is_some() {
                re_log::debug!("Overwritten table store with id: `{id}`");
            } else {
                re_log::debug!("Inserted table store with id: `{id}`");
            }
            self.command_sender
                .send_system(SystemCommand::set_selection(
                    re_viewer_context::Item::TableId(id),
                ));

            // If the viewer is in the background, tell the user that it has received something new.
            egui_ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
                egui::UserAttentionType::Informational,
            ));
        }
    }

    fn on_new_store(
        &self,
        egui_ctx: &egui::Context,
        store_id: &StoreId,
        channel_source: &LogSource,
        store_hub: &mut StoreHub,
    ) {
        if channel_source.select_when_loaded() {
            // Set the recording-id after potentially creating the store in the hub.
            // This ordering is important because the `StoreHub` internally
            // updates the app-id when changing the recording.
            match store_id.kind() {
                StoreKind::Recording => {
                    re_log::trace!("Opening a new recording: '{store_id:?}'");
                    self.make_store_active_and_highlight(store_hub, egui_ctx, store_id);
                }
                StoreKind::Blueprint => {
                    // We wait with activating blueprints until they are fully loaded,
                    // so that we don't run heuristics on half-loaded blueprints.
                    // Otherwise on a mixed connection (SDK sending both blueprint and recording)
                    // the blueprint won't be activated until the whole _recording_ has finished loading.
                }
            }
        }

        let entity_db = store_hub.entity_db_mut(store_id);
        let is_example = entity_db.store_class().is_example();

        if cfg!(target_arch = "wasm32") && !self.startup_options.is_in_notebook && !is_example {
            use std::sync::Once;
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                // Tell the user there is a faster native viewer they can use instead of the web viewer:
                let notification = re_ui::notifications::Notification::new(
                    re_ui::notifications::NotificationLevel::Tip, "For better performance, try the native Rerun Viewer!").with_link(
                    re_ui::Link {
                        text: "Install…".into(),
                        url: "https://rerun.io/docs/getting-started/installing-viewer#installing-the-viewer".into(),
                    }
                )
                    .no_toast()
                    .permanent_dismiss_id(egui::Id::new("install_native_viewer_prompt"));
                self.command_sender
                    .send_system(SystemCommand::ShowNotification(notification));
            });
        }

        if entity_db.store_kind() == StoreKind::Recording {
            #[cfg(feature = "analytics")]
            if let Some(analytics) = re_analytics::Analytics::global_or_init()
                && let Some(event) =
                    crate::viewer_analytics::event::open_recording(&self.app_env, entity_db)
            {
                analytics.record(event);
            }

            if let Some(event_dispatcher) = self.event_dispatcher.as_ref() {
                event_dispatcher.on_recording_open(entity_db);
            }
        }
    }

    fn receive_data_source_ui_command(
        &self,
        ui_command: DataSourceUiCommand,
        channel_source: &LogSource,
    ) {
        match ui_command {
            DataSourceUiCommand::SetUrlFragment { store_id, fragment } => {
                match re_uri::Fragment::from_str(&fragment) {
                    Ok(fragment) => {
                        self.command_sender
                            .send_system(SystemCommand::SetUrlFragment { store_id, fragment });
                    }

                    Err(err) => {
                        re_log::warn!(
                            "Failed to parse fragment received from {channel_source:?}: {err}"
                        );
                    }
                }
            }
        }
    }

    /// Makes the first recording store active that is found for a given data source if any.
    fn try_make_recording_from_source_active(
        &self,
        egui_ctx: &egui::Context,
        store_hub: &mut StoreHub,
        new_source: &LogSource,
    ) {
        if let Some(entity_db) = store_hub.find_recording_store_by_source(new_source) {
            let store_id = entity_db.store_id().clone();
            debug_assert!(store_id.is_recording()); // `find_recording_store_by_source` should have filtered for recordings rather than blueprints.
            self.make_store_active_and_highlight(store_hub, egui_ctx, &store_id);
        }
    }

    /// Makes the given store active and request user attention if Rerun in the background.
    fn make_store_active_and_highlight(
        &self,
        store_hub: &mut StoreHub,
        egui_ctx: &egui::Context,
        store_id: &StoreId,
    ) {
        if store_id.is_blueprint() {
            re_log::warn!(
                "Can't make a blueprint active: {store_id:?}. This is likely a bug in Rerun."
            );
            return;
        }

        store_hub.set_active_recording_id(store_id.clone());

        // Also select the new recording:
        self.command_sender
            .send_system(SystemCommand::set_selection(
                re_viewer_context::Item::StoreId(store_id.clone()),
            ));

        // If the viewer is in the background, tell the user that it has received something new.
        egui_ctx.send_viewport_cmd(egui::ViewportCommand::RequestUserAttention(
            egui::UserAttentionType::Informational,
        ));
    }

    /// After loading some data; check if the loaded data makes sense.
    fn validate_loaded_events(&self, store_events: &[re_chunk_store::ChunkStoreEvent]) {
        re_tracing::profile_function!();

        for event in store_events {
            let chunk = &event.diff.chunk;

            // For speed, we don't care about the order of the following log statements, so we silence this warning
            for component_descr in chunk.components().component_descriptors() {
                if let Some(archetype_name) = component_descr.archetype {
                    if let Some(archetype) = self.reflection.archetypes.get(&archetype_name) {
                        for &view_type in archetype.view_types {
                            if !cfg!(feature = "map_view") && view_type == "MapView" {
                                re_log::warn_once!(
                                    "Found map-related archetype, but viewer was not compiled with the `map_view` feature."
                                );
                            }
                        }
                    } else {
                        re_log::debug_once!("Unknown archetype: {archetype_name}");
                    }
                }
            }
        }
    }

    fn purge_memory_if_needed(&mut self, store_hub: &mut StoreHub) {
        re_tracing::profile_function!();

        fn format_limit(limit: Option<u64>) -> String {
            if let Some(bytes) = limit {
                format_bytes(bytes as _)
            } else {
                "∞".to_owned()
            }
        }

        use re_format::format_bytes;
        use re_memory::MemoryUse;

        let limit = self.startup_options.memory_limit;
        let mem_use_before = MemoryUse::capture();

        if let Some(minimum_fraction_to_purge) = limit.is_exceeded_by(&mem_use_before) {
            re_log::info_once!(
                "Reached memory limit of {}, dropping oldest data.",
                format_limit(limit.max_bytes)
            );

            let fraction_to_purge = (minimum_fraction_to_purge + 0.2).clamp(0.25, 1.0);

            re_log::trace!("RAM limit: {}", format_limit(limit.max_bytes));
            if let Some(resident) = mem_use_before.resident {
                re_log::trace!("Resident: {}", format_bytes(resident as _),);
            }
            if let Some(counted) = mem_use_before.counted {
                re_log::trace!("Counted: {}", format_bytes(counted as _));
            }

            re_tracing::profile_scope!("pruning");
            if let Some(counted) = mem_use_before.counted {
                re_log::trace!(
                    "Attempting to purge {:.1}% of used RAM ({})…",
                    100.0 * fraction_to_purge,
                    format_bytes(counted as f64 * fraction_to_purge as f64)
                );
            }

            let time_cursor_for =
                |store_id: &StoreId| -> Option<(re_log_types::Timeline, re_log_types::TimeInt)> {
                    let time_ctrl = self.state.time_controls.get(store_id)?;
                    Some((*time_ctrl.timeline()?, time_ctrl.time_int()?))
                };
            store_hub.purge_fraction_of_ram(fraction_to_purge, &time_cursor_for);

            let mem_use_after = MemoryUse::capture();

            let freed_memory = mem_use_before - mem_use_after;

            if let (Some(counted_before), Some(counted_diff)) =
                (mem_use_before.counted, freed_memory.counted)
                && 0 < counted_diff
            {
                re_log::debug!(
                    "Freed up {} ({:.1}%)",
                    format_bytes(counted_diff as _),
                    100.0 * counted_diff as f32 / counted_before as f32
                );
            }

            self.memory_panel.note_memory_purge();
        }
    }

    /// Reset the viewer to how it looked the first time you ran it.
    fn reset_viewer(&mut self, store_hub: &mut StoreHub, egui_ctx: &egui::Context) {
        self.state = Default::default();

        store_hub.clear_all_cloned_blueprints();

        // Reset egui:
        egui_ctx.memory_mut(|mem| *mem = Default::default());

        // Restore style:
        re_ui::apply_style_and_install_loaders(egui_ctx);

        if let Err(err) = crate::reset_viewer_persistence() {
            re_log::warn!("Failed to reset viewer: {err}");
        }
    }

    pub fn recording_db(&self) -> Option<&EntityDb> {
        self.store_hub.as_ref()?.active_recording()
    }

    // NOTE: Relying on `self` is dangerous, as this is called during a time where some internal
    // fields may have been temporarily `take()`n out. Keep this a static method.
    fn handle_dropping_files(
        egui_ctx: &egui::Context,
        storage_ctx: &StorageContext<'_>,
        command_sender: &CommandSender,
    ) {
        #![allow(clippy::allow_attributes, clippy::needless_continue)] // false positive, depending on target_arch

        preview_files_being_dropped(egui_ctx);

        let dropped_files = egui_ctx.input_mut(|i| std::mem::take(&mut i.raw.dropped_files));

        if dropped_files.is_empty() {
            return;
        }

        let mut force_store_info = false;

        for file in dropped_files {
            let active_store_id = storage_ctx
                .hub
                .active_store_id()
                .cloned()
                // Don't redirect data to the welcome screen.
                .filter(|store_id| store_id.application_id() != &StoreHub::welcome_screen_app_id())
                .unwrap_or_else(|| {
                    // When we're on the welcome screen, there is no recording ID to recommend.
                    // But we want one, otherwise multiple things being dropped simultaneously on the
                    // welcome screen would end up in different recordings!

                    // If we don't have any application ID to recommend (which means we are on the welcome screen),
                    // then we use the file path as the application ID or the file name if there is no path (on web builds).
                    let application_id = file
                        .path
                        .clone()
                        .map(|p| ApplicationId::from(p.display().to_string()))
                        .unwrap_or_else(|| ApplicationId::from(file.name.clone()));

                    // NOTE: We don't override blueprints' store IDs anyhow, so it is sound to assume that
                    // this can only be a recording.
                    let recording_id = RecordingId::random();

                    // We're creating a recording just-in-time, directly from the viewer.
                    // We need those store infos or the data will just be silently ignored.
                    force_store_info = true;

                    StoreId::recording(application_id, recording_id)
                });

            if let Some(bytes) = file.bytes {
                // This is what we get on Web.
                command_sender.send_system(SystemCommand::LoadDataSource(
                    LogDataSource::FileContents(
                        FileSource::DragAndDrop {
                            recommended_store_id: Some(active_store_id.clone()),
                            force_store_info,
                        },
                        FileContents {
                            name: file.name.clone(),
                            bytes: bytes.clone(),
                        },
                    ),
                ));

                continue;
            }

            #[cfg(not(target_arch = "wasm32"))]
            if let Some(path) = file.path {
                command_sender.send_system(SystemCommand::LoadDataSource(LogDataSource::FilePath(
                    FileSource::DragAndDrop {
                        recommended_store_id: Some(active_store_id.clone()),
                        force_store_info,
                    },
                    path,
                )));
            }
        }
    }

    fn should_fade_in_welcome_screen(&self) -> bool {
        if let Some(expect_data_soon) = self.startup_options.expect_data_soon {
            return expect_data_soon;
        }

        // The reason for the fade-in is to avoid the welcome screen
        // flickering quickly before receiving some data.
        // So: if we expect data very soon, we do a fade-in.

        for source in self.rx_log.sources() {
            match &*source {
                LogSource::File(_)
                | LogSource::RrdHttpStream { .. }
                | LogSource::RedapGrpcStream { .. }
                | LogSource::Stdin
                | LogSource::RrdWebEvent
                | LogSource::Sdk
                | LogSource::JsChannel { .. } => {
                    return true; // We expect data soon, so fade-in
                }

                LogSource::MessageProxy { .. } => {
                    // We start a gRPC server by default in native rerun, i.e. when just running `rerun`,
                    // and in that case fading in the welcome screen would be slightly annoying.
                    // However, we also use the gRPC server for sending data from the logging SDKs
                    // when they call `spawn()`, and in that case we really want to fade in the welcome screen.
                    // Therefore `spawn()` uses the special `--expect-data-soon` flag
                    // (handled earlier in this function), so here we know we are in the other case:
                    // a user calling `rerun` in their terminal (don't fade in).
                }
            }
        }

        false // No special sources (or no sources at all), so don't fade in
    }

    /// Handle fading in the welcome screen, if we should.
    fn welcome_screen_opacity(&self, egui_ctx: &egui::Context) -> f32 {
        if self.should_fade_in_welcome_screen() {
            // The reason for this delay is to avoid the welcome screen
            // flickering quickly before receiving some data.
            // The only time it has for that is between the call to `spawn` and sending the recording info,
            // which should happen _right away_, so we only need a small delay.
            // Why not skip the wlecome screen completely when we expect the data?
            // Because maybe the data never comes.
            let sec_since_first_shown = self.start_time.elapsed().as_secs_f32();
            let opacity = egui::remap_clamp(sec_since_first_shown, 0.4..=0.6, 0.0..=1.0);
            if opacity < 1.0 {
                egui_ctx.request_repaint();
            }
            opacity
        } else {
            1.0
        }
    }

    pub(crate) fn toggle_fullscreen(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let fullscreen = self
                .egui_ctx
                .input(|i| i.viewport().fullscreen.unwrap_or(false));
            self.egui_ctx
                .send_viewport_cmd(egui::ViewportCommand::Fullscreen(!fullscreen));
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(options) = &self.startup_options.fullscreen_options {
                // Tell JS to toggle fullscreen.
                if let Err(err) = options.on_toggle.call0() {
                    re_log::error!("{}", crate::web_tools::string_from_js_value(err));
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn is_fullscreen_allowed(&self) -> bool {
        self.startup_options.fullscreen_options.is_some()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn is_fullscreen_mode(&self) -> bool {
        if let Some(options) = &self.startup_options.fullscreen_options {
            // Ask JS if fullscreen is on or not.
            match options.get_state.call0() {
                Ok(v) => return v.is_truthy(),
                Err(err) => re_log::error_once!("{}", crate::web_tools::string_from_js_value(err)),
            }
        }

        false
    }

    #[allow(clippy::allow_attributes, clippy::needless_pass_by_ref_mut)] // False positive on wasm
    fn process_screenshot_result(
        &mut self,
        image: &Arc<egui::ColorImage>,
        user_data: &egui::UserData,
    ) {
        use re_viewer_context::ScreenshotInfo;

        if let Some(info) = user_data
            .data
            .as_ref()
            .and_then(|data| data.downcast_ref::<ScreenshotInfo>())
        {
            let ScreenshotInfo {
                ui_rect,
                pixels_per_point,
                name,
                target,
            } = (*info).clone();

            let rgba = if let Some(ui_rect) = ui_rect {
                Arc::new(image.region(&ui_rect, Some(pixels_per_point)))
            } else {
                image.clone()
            };

            match target {
                re_viewer_context::ScreenshotTarget::CopyToClipboard => {
                    self.egui_ctx.copy_image((*rgba).clone());
                }

                re_viewer_context::ScreenshotTarget::SaveToDisk => {
                    use image::ImageEncoder as _;
                    let mut png_bytes: Vec<u8> = Vec::new();
                    if let Err(err) = image::codecs::png::PngEncoder::new(&mut png_bytes)
                        .write_image(
                            rgba.as_raw(),
                            rgba.width() as u32,
                            rgba.height() as u32,
                            image::ExtendedColorType::Rgba8,
                        )
                    {
                        re_log::error!("Failed to encode screenshot as PNG: {err}");
                    } else {
                        let file_name = format!("{name}.png");
                        self.command_sender.save_file_dialog(
                            self.main_thread_token,
                            &file_name,
                            "Save screenshot".to_owned(),
                            png_bytes,
                        );
                    }
                }
            }
        } else {
            #[cfg(not(target_arch = "wasm32"))] // no full-app screenshotting on web
            self.screenshotter.save(&self.egui_ctx, image);
        }
    }

    /// Get a helper struct to interact with the given recording.
    pub fn blueprint_ctx<'a>(&'a self, recording_id: &StoreId) -> Option<AppBlueprintCtx<'a>> {
        let hub = self.store_hub.as_ref()?;

        let blueprint = hub.active_blueprint_for_app(recording_id.application_id())?;

        let default_blueprint = hub.default_blueprint_for_app(recording_id.application_id());

        let blueprint_query = self
            .state
            .get_blueprint_query_for_viewer(blueprint)
            .unwrap_or_else(|| {
                re_chunk::LatestAtQuery::latest(re_viewer_context::blueprint_timeline())
            });

        Some(AppBlueprintCtx {
            command_sender: &self.command_sender,
            current_blueprint: blueprint,
            default_blueprint,
            blueprint_query,
        })
    }
}

#[cfg(target_arch = "wasm32")]
fn blueprint_loader() -> BlueprintPersistence {
    // TODO(#2579): implement persistence for web
    BlueprintPersistence {
        loader: None,
        saver: None,
        validator: Some(Box::new(crate::blueprint::is_valid_blueprint)),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn blueprint_loader() -> BlueprintPersistence {
    use re_entity_db::StoreBundle;

    fn load_blueprint_from_disk(app_id: &ApplicationId) -> anyhow::Result<Option<StoreBundle>> {
        let blueprint_path = crate::saving::default_blueprint_path(app_id)?;
        if !blueprint_path.exists() {
            return Ok(None);
        }

        re_log::debug!("Trying to load blueprint for {app_id} from {blueprint_path:?}");

        if let Some(bundle) = crate::loading::load_blueprint_file(&blueprint_path) {
            for store in bundle.entity_dbs() {
                if store.store_kind() == StoreKind::Blueprint
                    && !crate::blueprint::is_valid_blueprint(store)
                {
                    re_log::warn_once!(
                        "Blueprint for {app_id} at {blueprint_path:?} appears invalid - will ignore. This is expected if you have just upgraded Rerun versions."
                    );
                    return Ok(None);
                }
            }
            Ok(Some(bundle))
        } else {
            Ok(None)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn save_blueprint_to_disk(app_id: &ApplicationId, blueprint: &EntityDb) -> anyhow::Result<()> {
        let blueprint_path = crate::saving::default_blueprint_path(app_id)?;

        let messages = blueprint.to_messages(None);
        let rrd_version = blueprint
            .store_info()
            .and_then(|info| info.store_version)
            .unwrap_or(re_build_info::CrateVersion::LOCAL);

        // TODO(jleibs): Should we push this into a background thread? Blueprints should generally
        // be small & fast to save, but maybe not once we start adding big pieces of user data?
        crate::saving::encode_to_file(rrd_version, &blueprint_path, messages)?;

        re_log::debug!("Saved blueprint for {app_id} to {blueprint_path:?}");

        Ok(())
    }

    BlueprintPersistence {
        loader: Some(Box::new(load_blueprint_from_disk)),
        saver: Some(Box::new(save_blueprint_to_disk)),
        validator: Some(Box::new(crate::blueprint::is_valid_blueprint)),
    }
}

impl eframe::App for App {
    fn clear_color(&self, visuals: &egui::Visuals) -> [f32; 4] {
        if re_ui::CUSTOM_WINDOW_DECORATIONS {
            [0.; 4] // transparent
        } else if visuals.dark_mode {
            [0., 0., 0., 1.]
        } else {
            [1., 1., 1., 1.]
        }
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if !self.startup_options.persist_state {
            return;
        }

        re_tracing::profile_function!();

        storage.set_string(RERUN_VERSION_KEY, self.build_info.version.to_string());

        // Save the app state
        eframe::set_value(storage, eframe::APP_KEY, &self.state);
        eframe::set_value(
            storage,
            REDAP_TOKEN_KEY,
            &self.connection_registry.dump_tokens(),
        );

        // Save the blueprints
        // TODO(#2579): implement web-storage for blueprints as well
        if let Some(hub) = &mut self.store_hub {
            if self.state.app_options.blueprint_gc {
                hub.gc_blueprints(&self.state.blueprint_undo_state);
            }

            if let Err(err) = hub.save_app_blueprints() {
                re_log::error!("Saving blueprints failed: {err}");
            }
        } else {
            re_log::error!("Could not save blueprints: the store hub is not available");
        }
    }

    fn update(&mut self, egui_ctx: &egui::Context, frame: &mut eframe::Frame) {
        #[cfg(all(not(target_arch = "wasm32"), feature = "perf_telemetry_tracy"))]
        re_perf_telemetry::external::tracing_tracy::client::frame_mark();

        if let Some(seconds) = frame.info().cpu_usage {
            self.frame_time_history
                .add(egui_ctx.input(|i| i.time), seconds);
        }

        #[cfg(target_arch = "wasm32")]
        if self.startup_options.enable_history {
            // Handle pressing the back/forward mouse buttons explicitly, since eframe catches those.
            let back_pressed =
                egui_ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Extra1));
            let fwd_pressed =
                egui_ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Extra2));

            if back_pressed {
                crate::web_history::go_back();
            }
            if fwd_pressed {
                crate::web_history::go_forward();
            }
        }

        // We move the time at the very start of the frame,
        // so that we always show the latest data when we're in "follow" mode.
        self.move_time();

        // Temporarily take the `StoreHub` out of the Viewer so it doesn't interfere with mutability
        let mut store_hub = self
            .store_hub
            .take()
            .expect("Failed to take store hub from the Viewer");

        // Update data source order so it's based on opening order.
        store_hub.update_data_source_order(&self.rx_log.sources());

        #[cfg(not(target_arch = "wasm32"))]
        if let Some(resolution_in_points) = self.startup_options.resolution_in_points.take() {
            egui_ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(
                resolution_in_points.into(),
            ));
        }

        #[cfg(not(target_arch = "wasm32"))]
        if self.screenshotter.update(egui_ctx).quit {
            egui_ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        if self.startup_options.memory_limit.is_unlimited() {
            // we only warn about high memory usage if the user hasn't specified a limit
            self.ram_limit_warner.update();
        }

        #[cfg(target_arch = "wasm32")]
        if let Some(PendingFilePromise {
            recommended_store_id,
            force_store_info,
            promise,
        }) = &self.open_files_promise
        {
            if let Some(files) = promise.ready() {
                for file in files {
                    self.command_sender
                        .send_system(SystemCommand::LoadDataSource(LogDataSource::FileContents(
                            FileSource::FileDialog {
                                recommended_store_id: recommended_store_id.clone(),
                                force_store_info: *force_store_info,
                            },
                            file.clone(),
                        )));
                }
                self.open_files_promise = None;
            }
        }

        // NOTE: GPU resource stats are cheap to compute so we always do.
        let gpu_resource_stats = {
            re_tracing::profile_scope!("gpu_resource_stats");

            let egui_renderer = frame
                .wgpu_render_state()
                .expect("Failed to get frame render state")
                .renderer
                .read();

            let render_ctx = egui_renderer
                .callback_resources
                .get::<re_renderer::RenderContext>()
                .expect("Failed to get render context");

            // Query statistics before begin_frame as this might be more accurate if there's resources that we recreate every frame.
            render_ctx.gpu_resources.statistics()
        };

        // NOTE: Store and caching stats are very costly to compute: only do so if the memory panel
        // is opened.
        let store_stats = self.memory_panel_open.then(|| store_hub.stats());

        // do early, before doing too many allocations
        self.memory_panel
            .update(&gpu_resource_stats, store_stats.as_ref());

        self.check_keyboard_shortcuts(egui_ctx);

        self.purge_memory_if_needed(&mut store_hub);

        // In some (rare) circumstances we run two egui passes in a single frame.
        // This happens on call to `egui::Context::request_discard`.
        let is_start_of_new_frame = egui_ctx.current_pass_index() == 0;
        if is_start_of_new_frame {
            // IMPORTANT: only call this once per FRAME even if we run multiple passes.
            // Otherwise we might incorrectly evict something that was invisible in the first (discarded) pass.
            store_hub.begin_frame_caches();
        }

        self.receive_messages(&mut store_hub, egui_ctx);

        if self.app_options().blueprint_gc {
            store_hub.gc_blueprints(&self.state.blueprint_undo_state);
        }

        store_hub.purge_empty();
        self.state.cleanup(&store_hub);

        file_saver_progress_ui(egui_ctx, &mut self.background_tasks); // toasts for background file saver
        #[cfg(not(target_arch = "wasm32"))]
        cropped_video_export_progress_ui(
            egui_ctx,
            &mut self.background_tasks,
            self.cropped_video_export.as_ref(),
        );

        // Make sure some app is active
        // Must be called before `read_context` below.
        if let DisplayMode::Loading(source) = self.state.navigation.current() {
            if !self.msg_receive_set().contains(source) {
                self.state.navigation.reset();
            }
        } else if store_hub.active_app().is_none() {
            let apps: std::collections::BTreeSet<&ApplicationId> = store_hub
                .store_bundle()
                .entity_dbs()
                .map(|db| db.application_id())
                .filter(|&app_id| app_id != &StoreHub::welcome_screen_app_id())
                .collect();
            if let Some(app_id) = apps.first().copied() {
                store_hub.set_active_app(app_id.clone());
                // set_active_app will also activate a new entry.
                // Select this entry so it's more obvious to the user which recording
                // is now active.
                match store_hub.active_recording_or_table() {
                    Some(RecordingOrTable::Recording { store_id }) => {
                        self.state
                            .selection_state
                            .set_selection(Item::StoreId(store_id.clone()));
                    }
                    Some(RecordingOrTable::Table { table_id }) => {
                        self.state
                            .selection_state
                            .set_selection(Item::TableId(table_id.clone()));
                    }
                    None => {}
                }
            } else {
                self.state.navigation.reset();
                store_hub.set_active_app(StoreHub::welcome_screen_app_id());
            }
        }

        {
            let (storage_context, store_context) = store_hub.read_context();

            let blueprint_query = store_context.as_ref().map_or_else(
                BlueprintUndoState::default_query,
                |store_context| {
                    self.state
                        .blueprint_query_for_viewer(store_context.blueprint)
                },
            );

            let app_blueprint = AppBlueprint::new(
                store_context.as_ref().map(|ctx| ctx.blueprint),
                &blueprint_query,
                egui_ctx,
                self.panel_state_overrides_active
                    .then_some(self.panel_state_overrides),
            );

            self.ui(
                egui_ctx,
                frame,
                &app_blueprint,
                &gpu_resource_stats,
                store_context.as_ref(),
                &storage_context,
                store_stats.as_ref(),
            );

            if re_ui::CUSTOM_WINDOW_DECORATIONS {
                // Paint the main window frame on top of everything else
                paint_native_window_frame(egui_ctx);
            }

            if let Some(cmd) = self.cmd_palette.show(
                egui_ctx,
                &crate::open_url_description::command_palette_parse_url,
            ) {
                match cmd {
                    re_ui::CommandPaletteAction::UiCommand(cmd) => {
                        self.command_sender.send_ui(cmd);
                    }
                    re_ui::CommandPaletteAction::OpenUrl(url_desc) => {
                        match url_desc.url.parse::<ViewerOpenUrl>() {
                            Ok(url) => {
                                url.open(
                                    egui_ctx,
                                    &OpenUrlOptions {
                                        follow_if_http: false,
                                        select_redap_source_when_loaded: true,
                                        show_loader: true,
                                    },
                                    &self.command_sender,
                                );
                            }
                            Err(err) => {
                                re_log::warn!("{err}");
                            }
                        }

                        // Note that we can't use `ui.ctx().open_url(egui::OpenUrl::same_tab(uri))` here because..
                        // * the url redirect in `check_for_clicked_hyperlinks` wouldn't be hit
                        // * we don't actually want to open any URLs in the browser here ever, only ever into the current viewer
                    }
                }
            }

            Self::handle_dropping_files(egui_ctx, &storage_context, &self.command_sender);

            // Run pending commands last (so we don't have to wait for a repaint before they are run):
            let display_mode = self.state.navigation.current().clone();
            self.run_pending_ui_commands(
                egui_ctx,
                &app_blueprint,
                &storage_context,
                store_context.as_ref(),
                &display_mode,
            );
        }
        self.run_pending_system_commands(&mut store_hub, egui_ctx);

        self.update_history(&store_hub);

        // Return the `StoreHub` to the Viewer so we have it on the next frame
        self.store_hub = Some(store_hub);

        {
            // Check for returned screenshots:
            let screenshots: Vec<_> = egui_ctx.input(|i| {
                i.raw
                    .events
                    .iter()
                    .filter_map(|event| {
                        if let egui::Event::Screenshot {
                            image, user_data, ..
                        } = event
                        {
                            Some((image.clone(), user_data.clone()))
                        } else {
                            None
                        }
                    })
                    .collect()
            });

            for (image, user_data) in screenshots {
                self.process_screenshot_result(&image, &user_data);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(&mut *self)
    }
}

fn paint_background_fill(ui: &egui::Ui) {
    // This is required because the streams view (time panel)
    // has rounded top corners, which leaves a gap.
    // So we fill in that gap (and other) here.
    // Of course this does some over-draw, but we have to live with that.

    let tokens = ui.tokens();

    ui.painter().rect_filled(
        ui.max_rect().shrink(0.5),
        tokens.native_window_corner_radius(),
        ui.visuals().panel_fill,
    );
}

fn paint_native_window_frame(egui_ctx: &egui::Context) {
    let tokens = egui_ctx.tokens();

    let painter = egui::Painter::new(
        egui_ctx.clone(),
        egui::LayerId::new(egui::Order::TOP, egui::Id::new("native_window_frame")),
        egui::Rect::EVERYTHING,
    );

    painter.rect_stroke(
        egui_ctx.content_rect(),
        tokens.native_window_corner_radius(),
        egui_ctx.tokens().native_frame_stroke,
        egui::StrokeKind::Inside,
    );
}

fn preview_files_being_dropped(egui_ctx: &egui::Context) {
    use egui::{Align2, Id, LayerId, Order, TextStyle};

    // Preview hovering files:
    if !egui_ctx.input(|i| i.raw.hovered_files.is_empty()) {
        use std::fmt::Write as _;

        let mut text = "Drop to load:\n".to_owned();
        egui_ctx.input(|input| {
            for file in &input.raw.hovered_files {
                if let Some(path) = &file.path {
                    write!(text, "\n{}", path.display()).ok();
                } else if !file.mime.is_empty() {
                    write!(text, "\n{}", file.mime).ok();
                }
            }
        });

        let painter =
            egui_ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("file_drop_target")));

        let screen_rect = egui_ctx.content_rect();
        painter.rect_filled(
            screen_rect,
            0.0,
            egui_ctx
                .style()
                .visuals
                .extreme_bg_color
                .gamma_multiply_u8(192),
        );
        painter.text(
            screen_rect.center(),
            Align2::CENTER_CENTER,
            text,
            TextStyle::Body.resolve(&egui_ctx.style()),
            egui_ctx.style().visuals.strong_text_color(),
        );
    }
}

// ----------------------------------------------------------------------------

fn file_saver_progress_ui(egui_ctx: &egui::Context, background_tasks: &mut BackgroundTasks) {
    if background_tasks.is_file_save_in_progress() {
        // There's already a file save running in the background.

        if let Some(res) = background_tasks.poll_file_saver_promise() {
            // File save promise has returned.
            match res {
                Ok(path) => {
                    re_log::info!("File saved to {path:?}."); // this will also show a notification the user
                }
                Err(err) => {
                    re_log::error!("{err}"); // this will also show a notification the user
                }
            }
        } else {
            // File save promise is still running in the background.

            // NOTE: not a toast, want something a bit more discreet here.
            egui::Window::new("file_saver_spin")
                .anchor(egui::Align2::RIGHT_BOTTOM, egui::Vec2::ZERO)
                .title_bar(false)
                .enabled(false)
                .auto_sized()
                .show(egui_ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Writing file to disk…");
                    })
                });
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cropped_video_export_progress_ui(
    egui_ctx: &egui::Context,
    background_tasks: &mut BackgroundTasks,
    export: Option<&CroppedVideoExport>,
) {
    if let Some(res) = background_tasks
        .poll_promise::<anyhow::Result<std::path::PathBuf>>(CROPPED_VIDEO_EXPORT_PROMISE)
    {
        match res {
            Ok(path) => {
                re_log::info!("Cropped video saved to {path:?}.");
            }
            Err(err) => {
                re_log::error!("{err}");
            }
        }
    }

    let Some((title, details)) = export
        .map(|export| {
            (
                "cropped_video_capture_spin",
                format!(
                    "Capturing cropped video… {}/{}",
                    export.frame_index + 1,
                    export.frame_times.len()
                ),
            )
        })
        .or_else(|| {
            background_tasks
                .is_promise_in_progress(CROPPED_VIDEO_EXPORT_PROMISE)
                .then(|| {
                    (
                        "cropped_video_encode_spin",
                        "Encoding cropped video…".to_owned(),
                    )
                })
        })
    else {
        return;
    };

    egui::Window::new(title)
        .anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(0.0, -36.0))
        .title_bar(false)
        .enabled(false)
        .auto_sized()
        .show(egui_ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(details);
            });
        });
}

/// [This may only be called on the main thread](https://docs.rs/rfd/latest/rfd/#macos-non-windowed-applications-async-and-threading).
#[cfg(not(target_arch = "wasm32"))]
fn open_file_dialog_native(_: crate::MainThreadToken) -> Vec<std::path::PathBuf> {
    re_tracing::profile_function!();

    let supported: Vec<_> = if re_data_loader::iter_external_loaders().len() == 0 {
        re_data_loader::supported_extensions().collect()
    } else {
        vec![]
    };

    let mut dialog = rfd::FileDialog::new();

    // If there's at least one external loader registered, then literally anything goes!
    if !supported.is_empty() {
        dialog = dialog.add_filter("Supported files", &supported);
    }

    dialog.pick_files().unwrap_or_default()
}

#[cfg(target_arch = "wasm32")]
async fn async_open_rrd_dialog() -> Vec<re_data_source::FileContents> {
    let supported: Vec<_> = re_data_loader::supported_extensions().collect();

    let files = rfd::AsyncFileDialog::new()
        .add_filter("Supported files", &supported)
        .pick_files()
        .await
        .unwrap_or_default();

    let mut file_contents = Vec::with_capacity(files.len());

    for file in files {
        let file_name = file.file_name();
        re_log::debug!("Reading {file_name}…");
        let bytes = file.read().await;
        re_log::debug!(
            "{file_name} was {}",
            re_format::format_bytes(bytes.len() as _)
        );
        file_contents.push(re_data_source::FileContents {
            name: file_name,
            bytes: bytes.into(),
        });
    }

    file_contents
}

#[cfg(not(target_arch = "wasm32"))]
fn save_cropped_video(
    app: &mut App,
    store_context: Option<&StoreContext<'_>>,
) -> anyhow::Result<()> {
    let Some(store_context) = store_context else {
        anyhow::bail!("No recording data to export");
    };

    if app.is_cropped_video_busy() {
        anyhow::bail!("A cropped video export is already in progress");
    }

    ensure_ffmpeg_available()?;

    let Some(export_area) = app.video_export_area(store_context) else {
        anyhow::bail!("No visible viewport layout is available for video export");
    };

    let time_ctrl = app
        .state
        .time_control(store_context.recording.store_id())
        .ok_or_else(|| anyhow::anyhow!("No time control available for the active recording"))?;

    let timeline = *time_ctrl.timeline_name();
    let Some(histogram) = store_context.recording.timeline_histograms().get(&timeline) else {
        anyhow::bail!("The active timeline has no temporal data");
    };

    let crop_range = app
        .state
        .loop_selection(Some(store_context))
        .filter(|(selection_timeline, _)| *selection_timeline == timeline)
        .map(|(_, range)| range.to_int())
        .unwrap_or_else(|| histogram.full_range());

    let frame_times = histogram
        .range(
            (
                Bound::Included(crop_range.min().as_i64()),
                Bound::Included(crop_range.max().as_i64()),
            ),
            1,
        )
        .map(|(range, _count)| TimeInt::new_temporal(range.min))
        .collect_vec();

    if frame_times.is_empty() {
        anyhow::bail!("The current crop range contains no frames on the active timeline");
    }

    let default_file_name = format!("{}.mp4", sanitize_file_name(&export_area.name));
    let Some(output_path) = rfd::FileDialog::new()
        .set_file_name(default_file_name)
        .set_title("Save video")
        .save_file()
    else {
        re_log::info!("No file selected - video not saved.");
        return Ok(());
    };

    let export_id = app.next_cropped_video_export_id;
    app.next_cropped_video_export_id += 1;

    let temp_dir = create_cropped_video_temp_dir(export_id)?;
    let fps = time_ctrl.fps().unwrap_or(30.0).max(1.0);

    let export = CroppedVideoExport {
        export_id,
        store_id: store_context.recording.store_id().clone(),
        original_timeline: *time_ctrl.timeline_name(),
        original_time: time_ctrl.time(),
        original_play_state: time_ctrl.play_state(),
        export_area,
        output_path,
        temp_dir,
        frame_times,
        frame_index: 0,
        frames_until_capture: 1,
        phase: CroppedVideoExportPhase::AwaitRenderedFrame,
        fps,
        captured_views: HashMap::default(),
    };

    let first_time = export.frame_times[0];
    app.command_sender
        .send_system(SystemCommand::TimeControlCommands {
            store_id: export.store_id.clone(),
            time_commands: vec![
                TimeControlCommand::SetActiveTimeline(timeline),
                TimeControlCommand::SetPlayState(PlayState::Paused),
                TimeControlCommand::SetTime(first_time.into()),
            ],
        });

    re_log::info!(
        "Starting video export for {:?} with {} frame(s).",
        export.export_area.name,
        export.frame_times.len()
    );

    app.cropped_video_export = Some(export);
    app.egui_ctx.request_repaint();

    Ok(())
}

fn save_active_recording(
    app: &mut App,
    store_context: Option<&StoreContext<'_>>,
    loop_selection: Option<(TimelineName, re_log_types::AbsoluteTimeRangeF)>,
) -> anyhow::Result<()> {
    let Some(entity_db) = store_context.as_ref().map(|view| view.recording) else {
        // NOTE: Can only happen if saving through the command palette.
        anyhow::bail!("No recording data to save");
    };

    save_recording(app, entity_db, loop_selection)
}

#[cfg(not(target_arch = "wasm32"))]
impl OffscreenViewportRenderer {
    fn new(
        frame_render_state: &egui_wgpu::RenderState,
        live_egui_ctx: &egui::Context,
    ) -> anyhow::Result<Self> {
        let egui_ctx = egui::Context::default();
        egui_ctx.set_theme(live_egui_ctx.theme());
        egui_ctx.set_os(live_egui_ctx.os());
        egui_ctx.set_style((*live_egui_ctx.style()).clone());
        re_ui::apply_style_and_install_loaders(&egui_ctx);

        let mut renderer = egui_wgpu::Renderer::new(
            &frame_render_state.device,
            frame_render_state.target_format,
            egui_wgpu::RendererOptions::default(),
        );
        renderer
            .callback_resources
            .insert(re_renderer::RenderContext::new(
                &frame_render_state.adapter,
                frame_render_state.device.clone(),
                frame_render_state.queue.clone(),
                frame_render_state.target_format,
                re_renderer::RenderConfig::best_for_device_caps,
            )?);

        Ok(Self {
            egui_ctx,
            renderer,
            device: frame_render_state.device.clone(),
            queue: frame_render_state.queue.clone(),
            target_format: frame_render_state.target_format,
        })
    }

    fn render_ctx(&self) -> &re_renderer::RenderContext {
        self.renderer
            .callback_resources
            .get::<re_renderer::RenderContext>()
            .expect("offscreen renderer is missing RenderContext")
    }

    fn render_ctx_mut(&mut self) -> &mut re_renderer::RenderContext {
        self.renderer
            .callback_resources
            .get_mut::<re_renderer::RenderContext>()
            .expect("offscreen renderer is missing RenderContext")
    }

    fn render_viewport_frame(
        &mut self,
        size_in_points: egui::Vec2,
        pixels_per_point: f32,
        run_ui: impl FnOnce(&egui::Context, &re_renderer::RenderContext),
    ) -> anyhow::Result<image::RgbaImage> {
        self.render_ctx_mut().begin_frame();

        let mut input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size_in_points)),
            ..Default::default()
        };
        let viewport = input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap();
        viewport.native_pixels_per_point = Some(pixels_per_point);

        let render_ctx = self.render_ctx();
        let mut run_ui = Some(run_ui);
        let output = self.egui_ctx.run(input, |ctx| {
            run_ui
                .take()
                .expect("offscreen viewport frame should render once")(ctx, render_ctx)
        });
        self.render_ctx_mut().before_submit();

        for (id, image) in &output.textures_delta.set {
            self.renderer
                .update_texture(&self.device, &self.queue, *id, image);
        }

        let size = size_in_points * pixels_per_point;
        let screen = egui_wgpu::ScreenDescriptor {
            pixels_per_point,
            size_in_pixels: [size.x.round() as u32, size.y.round() as u32],
        };
        let tessellated = self
            .egui_ctx
            .tessellate(output.shapes.clone(), pixels_per_point);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Offscreen video export encoder"),
            });

        let user_buffers = self.renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tessellated,
            &screen,
        );

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen video export texture"),
            size: wgpu::Extent3d {
                width: screen.size_in_pixels[0],
                height: screen.size_in_pixels[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Offscreen video export render pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    ..Default::default()
                })
                .forget_lifetime();

            self.renderer.render(&mut pass, &tessellated, &screen);
        }

        self.queue
            .submit(user_buffers.into_iter().chain(once(encoder.finish())));
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(OFFSCREEN_EXPORT_WAIT_TIMEOUT),
        })?;

        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }

        read_texture_to_image(&self.device, &self.queue, &texture, self.target_format)
    }
}

fn save_recording(
    app: &mut App,
    entity_db: &EntityDb,
    loop_selection: Option<(TimelineName, re_log_types::AbsoluteTimeRangeF)>,
) -> anyhow::Result<()> {
    let rrd_version = entity_db
        .store_info()
        .and_then(|info| info.store_version)
        .unwrap_or(re_build_info::CrateVersion::LOCAL);

    let file_name = if let Some(recording_name) = entity_db
        .recording_info_property::<re_sdk_types::components::Name>(
            re_sdk_types::archetypes::RecordingInfo::descriptor_name().component,
        ) {
        format!("{}.rrd", sanitize_file_name(&recording_name))
    } else {
        "data.rrd".to_owned()
    };

    let title = if loop_selection.is_some() {
        "Save cropped recording"
    } else {
        "Save recording"
    };

    save_entity_db(
        app,
        rrd_version,
        file_name,
        title.to_owned(),
        entity_db.to_messages(loop_selection),
    )
}

fn save_blueprint(app: &mut App, store_context: Option<&StoreContext<'_>>) -> anyhow::Result<()> {
    let Some(store_context) = store_context else {
        anyhow::bail!("No blueprint to save");
    };

    re_tracing::profile_function!();

    let rrd_version = store_context
        .blueprint
        .store_info()
        .and_then(|info| info.store_version)
        .unwrap_or(re_build_info::CrateVersion::LOCAL);

    // We change the recording id to a new random one,
    // otherwise when saving and loading a blueprint file, we can end up
    // in a situation where the store_id we're loading is the same as the currently active one,
    // which mean they will merge in a strange way.
    // This is also related to https://github.com/rerun-io/rerun/issues/5295
    let new_store_id = store_context
        .blueprint
        .store_id()
        .clone()
        .with_recording_id(RecordingId::random());
    let messages = store_context.blueprint.to_messages(None).map(|mut msg| {
        if let Ok(msg) = &mut msg {
            msg.set_store_id(new_store_id.clone());
        }
        msg
    });

    let file_name = format!(
        "{}.rbl",
        crate::saving::sanitize_app_id(store_context.application_id())
    );
    let title = "Save blueprint";

    save_entity_db(app, rrd_version, file_name, title.to_owned(), messages)
}

// TODO(emilk): unify this with `ViewerContext::save_file_dialog`
#[allow(clippy::allow_attributes, clippy::needless_pass_by_ref_mut)] // `app` is only used on native
#[allow(clippy::allow_attributes, clippy::unnecessary_wraps)] // cannot return error on web
fn save_entity_db(
    #[allow(clippy::allow_attributes, unused_variables)] app: &mut App, // only used on native
    rrd_version: CrateVersion,
    file_name: String,
    title: String,
    messages: impl Iterator<Item = re_chunk::ChunkResult<LogMsg>>,
) -> anyhow::Result<()> {
    re_tracing::profile_function!();

    // TODO(#6984): Ideally we wouldn't collect at all and just stream straight to the
    // encoder from the store.
    //
    // From a memory usage perspective this isn't too bad though: the data within is still
    // refcounted straight from the store in any case.
    //
    // It just sucks latency-wise.
    let messages = messages.collect::<Vec<_>>();

    // Web
    #[cfg(target_arch = "wasm32")]
    {
        wasm_bindgen_futures::spawn_local(async move {
            if let Err(err) =
                async_save_dialog(rrd_version, &file_name, &title, messages.into_iter()).await
            {
                re_log::error!("File saving failed: {err}");
            }
        });
    }

    // Native
    #[cfg(not(target_arch = "wasm32"))]
    {
        let path = {
            re_tracing::profile_scope!("file_dialog");
            rfd::FileDialog::new()
                .set_file_name(file_name)
                .set_title(title)
                .save_file()
        };
        if let Some(path) = path {
            app.background_tasks.spawn_file_saver(move || {
                crate::saving::encode_to_file(rrd_version, &path, messages.into_iter())?;
                Ok(path)
            })?;
        }
    }

    Ok(())
}

#[cfg(target_arch = "wasm32")]
async fn async_save_dialog(
    rrd_version: CrateVersion,
    file_name: &str,
    title: &str,
    messages: impl Iterator<Item = re_chunk::ChunkResult<LogMsg>>,
) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let file_handle = rfd::AsyncFileDialog::new()
        .set_file_name(file_name)
        .set_title(title)
        .save_file()
        .await;

    let Some(file_handle) = file_handle else {
        return Ok(()); // aborted
    };

    let options = re_log_encoding::rrd::EncodingOptions::PROTOBUF_COMPRESSED;
    let mut bytes = Vec::new();
    re_log_encoding::Encoder::encode_into(rrd_version, options, messages, &mut bytes)?;
    file_handle.write(&bytes).await.context("Failed to save")
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure_ffmpeg_available() -> anyhow::Result<()> {
    let output = std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output();

    match output {
        Ok(output) if output.status.success() => Ok(()),
        Ok(_) => anyhow::bail!("`ffmpeg` is installed but unavailable for video export"),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            anyhow::bail!("`ffmpeg` was not found on PATH. Install ffmpeg to export MP4 videos.")
        }
        Err(err) => Err(err.into()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn create_cropped_video_temp_dir(export_id: u64) -> anyhow::Result<std::path::PathBuf> {
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = std::env::temp_dir().join(format!(
        "rerun-cropped-video-{}-{}-{}",
        std::process::id(),
        export_id,
        millis
    ));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn cropped_video_frame_path(temp_dir: &std::path::Path, frame_index: usize) -> std::path::PathBuf {
    temp_dir.join(format!("frame_{frame_index:06}.png"))
}

#[cfg(not(target_arch = "wasm32"))]
fn write_cropped_video_frame_rgba(
    temp_dir: &std::path::Path,
    frame_index: usize,
    image: &image::RgbaImage,
) -> anyhow::Result<()> {
    let path = cropped_video_frame_path(temp_dir, frame_index);
    image.save(&path)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn composite_cropped_video_frame(
    export: &mut CroppedVideoExport,
    background_fill: egui::Color32,
) -> anyhow::Result<image::RgbaImage> {
    let width = ((export.export_area.rect.width() * export.export_area.pixels_per_point).round()
        as u32)
        .max(1);
    let height = ((export.export_area.rect.height() * export.export_area.pixels_per_point).round()
        as u32)
        .max(1);

    let mut composited =
        image::RgbaImage::from_pixel(width, height, image::Rgba(background_fill.to_array()));

    for view in &export.export_area.views {
        let Some(view_frame) = export.captured_views.get(&view.view_id) else {
            anyhow::bail!("Missing captured view image for {:?}", view.view_id);
        };

        let min =
            (view_frame.ui_rect.min - export.export_area.rect.min) * view_frame.pixels_per_point;
        image::imageops::overlay(
            &mut composited,
            &view_frame.image,
            min.x.round() as i64,
            min.y.round() as i64,
        );
    }

    Ok(composited)
}

#[cfg(not(target_arch = "wasm32"))]
fn is_renderer_video_export_supported(class_identifier: re_sdk_types::ViewClassIdentifier) -> bool {
    class_identifier == SpatialView2D::identifier()
        || class_identifier == SpatialView3D::identifier()
        || class_identifier == MapView::identifier()
}

#[cfg(not(target_arch = "wasm32"))]
fn read_texture_to_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    texture_format: wgpu::TextureFormat,
) -> anyhow::Result<image::RgbaImage> {
    let width = texture.width() as usize;
    let height = texture.height() as usize;
    let unpadded_bytes_per_row = width * std::mem::size_of::<u32>();
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) / align * align;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Offscreen video export readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Offscreen video export readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: texture.width(),
            height: texture.height(),
            depth_or_array_layers: 1,
        },
    );

    let submission_index = queue.submit(once(encoder.finish()));
    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: Some(OFFSCREEN_EXPORT_WAIT_TIMEOUT),
    })?;
    receiver.recv()?.map_err(anyhow::Error::from)?;

    let data = output_buffer
        .slice(..)
        .get_mapped_range()
        .chunks_exact(padded_bytes_per_row)
        .flat_map(|row| row.iter().take(unpadded_bytes_per_row))
        .copied()
        .collect::<Vec<_>>();

    let mut image = image::RgbaImage::from_raw(texture.width(), texture.height(), data)
        .ok_or_else(|| anyhow::anyhow!("Failed to convert offscreen texture to RGBA image"))?;

    if texture_format == wgpu::TextureFormat::Bgra8Unorm
        || texture_format == wgpu::TextureFormat::Bgra8UnormSrgb
    {
        for pixel in image.pixels_mut() {
            pixel.0.swap(0, 2);
        }
    }

    Ok(image)
}

#[cfg(not(target_arch = "wasm32"))]
fn encode_cropped_video_frames(
    temp_dir: &std::path::Path,
    output_path: &std::path::Path,
    fps: f32,
) -> anyhow::Result<std::path::PathBuf> {
    let input_pattern = temp_dir.join("frame_%06d.png");

    let output = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg(&input_pattern)
        .arg("-vf")
        .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:v")
        .arg("libx264")
        .arg(output_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "ffmpeg failed to encode {:?}: {stderr}\nTemporary frames kept at {:?}",
            output_path,
            temp_dir
        );
    }

    if let Err(err) = std::fs::remove_dir_all(temp_dir)
        && err.kind() != std::io::ErrorKind::NotFound
    {
        re_log::debug!(
            "Failed to clean up temporary video frames at {:?}: {err}",
            temp_dir
        );
    }

    Ok(output_path.to_path_buf())
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_streaming_video_encoder(
    output_path: &std::path::Path,
    fps: f32,
    width: u32,
    height: u32,
) -> anyhow::Result<std::process::Child> {
    Ok(std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-f")
        .arg("rawvideo")
        .arg("-pixel_format")
        .arg("rgba")
        .arg("-video_size")
        .arg(format!("{width}x{height}"))
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg("-")
        .arg("-vf")
        .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-c:v")
        .arg("libx264")
        .arg(output_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()?)
}

#[cfg(not(target_arch = "wasm32"))]
fn write_video_frame_to_ffmpeg(
    ffmpeg: &mut std::process::Child,
    image: &image::RgbaImage,
) -> anyhow::Result<()> {
    let stdin = ffmpeg
        .stdin
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("ffmpeg stdin was not available for video export"))?;
    stdin.write_all(image.as_raw())?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn finish_streaming_video_encoder(
    mut ffmpeg: std::process::Child,
    output_path: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf> {
    drop(ffmpeg.stdin.take());
    let output = ffmpeg.wait_with_output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed to encode {:?}: {stderr}", output_path);
    }

    Ok(output_path.to_path_buf())
}

/// Propagates [`re_viewer_context::TimeControlResponse`] to [`ViewerEventDispatcher`].
fn handle_time_ctrl_event(
    recording: &EntityDb,
    events: Option<&ViewerEventDispatcher>,
    response: &re_viewer_context::TimeControlResponse,
) {
    let Some(events) = events else {
        return;
    };

    if let Some(playing) = response.playing_change {
        events.on_play_state_change(recording, playing);
    }

    if let Some((timeline, time)) = response.timeline_change {
        events.on_timeline_change(recording, timeline, time);
    }

    if let Some(time) = response.time_change {
        events.on_time_update(recording, time);
    }
}
