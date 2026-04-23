use ahash::HashSet;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::ViewId;

slotmap::new_key_type! { pub struct ViewBuilderHandle; }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RendererPaneMetadata {
    pub view_id: ViewId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RendererVideoExportScreenshot {
    pub export_id: u64,
    pub frame_index: usize,
    pub view_id: ViewId,
    pub ui_rect: egui::Rect,
    pub pixels_per_point: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RendererVideoExportRequest {
    export_id: u64,
    frame_index: usize,
    remaining_view_ids: HashSet<ViewId>,
}

#[derive(Default)]
pub struct RendererVideoExportState {
    current_request: Mutex<Option<RendererVideoExportRequest>>,
}

impl RendererVideoExportState {
    pub fn set_request(
        &self,
        export_id: u64,
        frame_index: usize,
        view_ids: impl IntoIterator<Item = ViewId>,
    ) {
        let remaining_view_ids = view_ids.into_iter().collect();
        *self.current_request.lock() = Some(RendererVideoExportRequest {
            export_id,
            frame_index,
            remaining_view_ids,
        });
    }

    pub fn clear_request(&self) {
        self.current_request.lock().take();
    }

    pub fn take_request_for_view(
        &self,
        view_id: ViewId,
        ui_rect: egui::Rect,
        pixels_per_point: f32,
    ) -> Option<RendererVideoExportScreenshot> {
        let mut current_request = self.current_request.lock();
        let request = current_request.as_mut()?;

        if !request.remaining_view_ids.remove(&view_id) {
            return None;
        }

        let screenshot = RendererVideoExportScreenshot {
            export_id: request.export_id,
            frame_index: request.frame_index,
            view_id,
            ui_rect,
            pixels_per_point,
        };

        if request.remaining_view_ids.is_empty() {
            current_request.take();
        }

        Some(screenshot)
    }
}

pub fn new_renderer_callback(
    view_builder: re_renderer::ViewBuilder,
    viewport: egui::Rect,
    clear_color: re_renderer::Rgba,
    pane_metadata: Option<RendererPaneMetadata>,
) -> egui::PaintCallback {
    egui_wgpu::Callback::new_paint_callback(
        viewport,
        ReRendererCallback {
            view_builder: Mutex::new(view_builder),
            clear_color,
            viewport,
            pane_metadata,
        },
    )
}

struct ReRendererCallback {
    view_builder: Mutex<re_renderer::ViewBuilder>,
    clear_color: re_renderer::Rgba,
    viewport: egui::Rect,
    pane_metadata: Option<RendererPaneMetadata>,
}

impl egui_wgpu::CallbackTrait for ReRendererCallback {
    // TODO(andreas): Prepare callbacks should run in parallel.
    //                Command buffer recording may be fairly expensive in the future!
    //                Sticking to egui's current model, each prepare callback could fork of a task and in finish_prepare we wait for them.
    fn prepare(
        &self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        paint_callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(ctx) = paint_callback_resources.get::<re_renderer::RenderContext>() else {
            re_log::error_once!(
                "Failed to execute egui prepare callback. No render context available."
            );
            return Vec::new();
        };

        if let Some(metadata) = self.pane_metadata
            && let Some(export_state) =
                paint_callback_resources.get::<Arc<RendererVideoExportState>>()
            && let Some(request) = export_state.take_request_for_view(
                metadata.view_id,
                self.viewport,
                _screen_descriptor.pixels_per_point,
            )
            && let Err(err) =
                self.view_builder
                    .lock()
                    .schedule_screenshot(ctx, request.export_id, request)
        {
            re_log::warn_once!("Failed to schedule video export screenshot: {err}");
        }

        match self.view_builder.lock().draw(ctx, self.clear_color) {
            Ok(command_buffer) => vec![command_buffer],
            Err(err) => {
                re_log::error_once!("Failed to fill view builder: {err}");
                // TODO(andreas): It would be nice to paint an error message instead.
                Vec::new()
            }
        }
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        paint_callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(ctx) = paint_callback_resources.get::<re_renderer::RenderContext>() else {
            // TODO(#4433): Shouldn't show up like this.
            re_log::error_once!(
                "Failed to execute egui draw callback. No render context available."
            );
            return;
        };
        self.view_builder.lock().composite(ctx, render_pass);
    }
}
