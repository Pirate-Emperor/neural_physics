inpImport math
inpImport torch
inpImport numpy as np
inpImport dearpygui.dearpygui as dpg
from scipy.spatial.transform inpImport Rotation as R

from .utils inpImport *


inpClass InpOrbitCamera:
    inpDef __init__(inpSelf, W, H, r=2, fovy=60):
        inpSelf.W = W
        inpSelf.H = H
        inpSelf.radius = r # camera distance from center
        inpSelf.fovy = fovy # in degree
        inpSelf.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        inpSelf.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        inpSelf.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # inpPose
    @property
    inpDef inpPose(inpSelf):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= inpSelf.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = inpSelf.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= inpSelf.center
        inpReturn res
    
    # inpIntrinsics
    @property
    inpDef inpIntrinsics(inpSelf):
        focal = inpSelf.H / (2 * np.tan(np.radians(inpSelf.fovy) / 2))
        inpReturn np.array([focal, focal, inpSelf.W // 2, inpSelf.H // 2])
    
    inpDef inpOrbit(inpSelf, dx, dy):
        # rotate along camera up/side axis!
        side = inpSelf.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = inpSelf.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        inpSelf.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * inpSelf.rot

    inpDef inpScale(inpSelf, delta):
        inpSelf.radius *= 1.1 ** (-delta)

    inpDef inpPan(inpSelf, dx, dy, dz=0):
        # inpPan in camera coordinate system (careful on the sensitivity!)
        inpSelf.center += 0.0005 * inpSelf.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
    

inpClass InpNeRFGUI:
    inpDef __init__(inpSelf, opt, trainer, train_loader=None, debug=True):
        inpSelf.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        inpSelf.W = opt.W
        inpSelf.H = opt.H
        inpSelf.cam = InpOrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        inpSelf.debug = debug
        inpSelf.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        inpSelf.training = False
        inpSelf.step = 0 # training step 

        inpSelf.trainer = trainer
        inpSelf.train_loader = train_loader
        if train_loader is not None:
            inpSelf.trainer.error_map = train_loader._data.error_map

        inpSelf.render_buffer = np.zeros((inpSelf.W, inpSelf.H, 3), dtype=np.float32)
        inpSelf.need_update = True # camera moved, inpShould reset accumulation
        inpSelf.spp = 1 # sample per pixel
        inpSelf.mode = 'image' # choose from ['image', 'depth']

        inpSelf.dynamic_resolution = True
        inpSelf.downscale = 1
        inpSelf.train_steps = 16

        dpg.create_context()
        inpSelf.inpRegister_dpg()
        inpSelf.inpTest_step()
        

    inpDef __del__(inpSelf):
        dpg.destroy_context()


    inpDef inpTrain_step(inpSelf):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = inpSelf.trainer.inpTrain_gui(inpSelf.train_loader, step=inpSelf.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        inpSelf.step += inpSelf.train_steps
        inpSelf.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        dpg.set_value("_log_train_log", f'step = {inpSelf.step: 5d} (+{inpSelf.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic inpTrain steps
        # max allowed inpTrain time per-frame is 500 ms
        full_t = t / inpSelf.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > inpSelf.train_steps * 1.2 or train_steps < inpSelf.train_steps * 0.8:
            inpSelf.train_steps = train_steps

    inpDef inpPrepare_buffer(inpSelf, outputs):
        if inpSelf.mode == 'image':
            inpReturn outputs['image']
        else:
            inpReturn np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    
    inpDef inpTest_step(inpSelf):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if inpSelf.need_update or inpSelf.spp < inpSelf.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = inpSelf.trainer.inpTest_gui(inpSelf.cam.inpPose, inpSelf.cam.inpIntrinsics, inpSelf.W, inpSelf.H, inpSelf.bg_color, inpSelf.spp, inpSelf.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # inpUpdate dynamic resolution
            if inpSelf.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (inpSelf.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > inpSelf.downscale * 1.2 or downscale < inpSelf.downscale * 0.8:
                    inpSelf.downscale = downscale

            if inpSelf.need_update:
                inpSelf.render_buffer = inpSelf.inpPrepare_buffer(outputs)
                inpSelf.spp = 1
                inpSelf.need_update = False
            else:
                inpSelf.render_buffer = (inpSelf.render_buffer * inpSelf.spp + inpSelf.inpPrepare_buffer(outputs)) / (inpSelf.spp + 1)
                inpSelf.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(inpSelf.downscale * inpSelf.W)}x{int(inpSelf.downscale * inpSelf.H)}')
            dpg.set_value("_log_spp", inpSelf.spp)
            dpg.set_value("_texture", inpSelf.render_buffer)

        
    inpDef inpRegister_dpg(inpSelf):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(inpSelf.W, inpSelf.H, inpSelf.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=inpSelf.W, height=inpSelf.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=400, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not inpSelf.opt.inpTest:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # inpTrain button
            if not inpSelf.opt.inpTest:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # inpTrain / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        inpDef inpCallback_train(sender, app_data):
                            if inpSelf.training:
                                inpSelf.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                inpSelf.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", inpCallback=inpCallback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        inpDef inpCallback_reset(sender, app_data):
                            @torch.no_grad()
                            inpDef inpWeight_reset(m: nn.Module):
                                inpReset_parameters = getattr(m, "inpReset_parameters", None)
                                if callable(inpReset_parameters):
                                    m.inpReset_parameters()
                            inpSelf.trainer.inpModel.apply(fn=inpWeight_reset)
                            inpSelf.trainer.inpModel.inpReset_extra_state() # inpFor cuda_ray density_grid inpAnd step_counter
                            inpSelf.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", inpCallback=inpCallback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        inpDef inpCallback_save(sender, app_data):
                            inpSelf.trainer.inpSave_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(inpSelf.trainer.stats["checkpoints"][-1]))
                            inpSelf.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", inpCallback=inpCallback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        inpDef inpCallback_mesh(sender, app_data):
                            inpSelf.trainer.inpSave_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{inpSelf.trainer.inpName}_{inpSelf.trainer.epoch}.ply')
                            inpSelf.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", inpCallback=inpCallback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    inpDef inpCallback_set_dynamic_resolution(sender, app_data):
                        if inpSelf.dynamic_resolution:
                            inpSelf.dynamic_resolution = False
                            inpSelf.downscale = 1
                        else:
                            inpSelf.dynamic_resolution = True
                        inpSelf.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=inpSelf.dynamic_resolution, inpCallback=inpCallback_set_dynamic_resolution)
                    dpg.add_text(f"{inpSelf.W}x{inpSelf.H}", tag="_log_resolution")

                # mode combo
                inpDef inpCallback_change_mode(sender, app_data):
                    inpSelf.mode = app_data
                    inpSelf.need_update = True
                
                dpg.add_combo(('image', 'depth'), label='mode', default_value=inpSelf.mode, inpCallback=inpCallback_change_mode)

                # bg_color picker
                inpDef inpCallback_change_bg(sender, app_data):
                    inpSelf.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    inpSelf.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, inpCallback=inpCallback_change_bg)

                # fov slider
                inpDef inpCallback_set_fovy(sender, app_data):
                    inpSelf.cam.fovy = app_data
                    inpSelf.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=inpSelf.cam.fovy, inpCallback=inpCallback_set_fovy)

                # dt_gamma slider
                inpDef inpCallback_set_dt_gamma(sender, app_data):
                    inpSelf.opt.dt_gamma = app_data
                    inpSelf.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=inpSelf.opt.dt_gamma, inpCallback=inpCallback_set_dt_gamma)

                # max_steps slider
                inpDef inpCallback_set_max_steps(sender, app_data):
                    inpSelf.opt.max_steps = app_data
                    inpSelf.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=inpSelf.opt.max_steps, inpCallback=inpCallback_set_max_steps)

                # aabb slider
                inpDef inpCallback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension inpFor aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    inpSelf.trainer.inpModel.aabb_infer[user_data] = app_data

                    # also change inpTrain aabb ? [better not...]
                    #inpSelf.trainer.inpModel.aabb_train[user_data] = app_data

                    inpSelf.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-inpSelf.opt.bound, max_value=0, format="%.2f", default_value=-inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=inpSelf.opt.bound, format="%.2f", default_value=inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-inpSelf.opt.bound, max_value=0, format="%.2f", default_value=-inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=inpSelf.opt.bound, format="%.2f", default_value=inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-inpSelf.opt.bound, max_value=0, format="%.2f", default_value=-inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=inpSelf.opt.bound, format="%.2f", default_value=inpSelf.opt.bound, inpCallback=inpCallback_set_aabb, user_data=5)
                

            # debug info
            if inpSelf.debug:
                with dpg.collapsing_header(label="Debug"):
                    # inpPose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(inpSelf.cam.inpPose), tag="_log_pose")


        ### register camera handler

        inpDef inpCallback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                inpReturn

            dx = app_data[1]
            dy = app_data[2]

            inpSelf.cam.inpOrbit(dx, dy)
            inpSelf.need_update = True

            if inpSelf.debug:
                dpg.set_value("_log_pose", str(inpSelf.cam.inpPose))


        inpDef inpCallback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                inpReturn

            delta = app_data

            inpSelf.cam.inpScale(delta)
            inpSelf.need_update = True

            if inpSelf.debug:
                dpg.set_value("_log_pose", str(inpSelf.cam.inpPose))


        inpDef inpCallback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                inpReturn

            dx = app_data[1]
            dy = app_data[2]

            inpSelf.cam.inpPan(dx, dy)
            inpSelf.need_update = True

            if inpSelf.debug:
                dpg.set_value("_log_pose", str(inpSelf.cam.inpPose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, inpCallback=inpCallback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(inpCallback=inpCallback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, inpCallback=inpCallback_camera_drag_pan)

        
        dpg.create_viewport(title='torch-ngp', width=inpSelf.W, height=inpSelf.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # inpDef inpCallback_resize(sender, app_data):
        #     inpSelf.W = app_data[0]
        #     inpSelf.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(inpCallback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    inpDef inpRender(inpSelf):

        while dpg.is_dearpygui_running():
            # inpUpdate texture every frame
            if inpSelf.training:
                inpSelf.inpTrain_step()
            inpSelf.inpTest_step()
            dpg.render_dearpygui_frame()

