import os
import glob
import numpy as np

from .math import rotation_to_euler, euler_to_rotation

import bpy


class BlenderSceneRenderer():


    def __init__(self, params):

        self.width            = params["width"]
        self.height           = params["height"]
        self.z_min            = params["zmin"]
        self.z_max            = params["zmax"]
        self.focal            = params["focal"]
        self.colormode        = params["colormode"]
        self.colordepth       = str(params["colordepth"])
        self.outputext        = params["outputext"]
        self.outputfolder     = "."
        self.add_ground_plane = params["add_ground_plane"]
        self.photorealism     = params["photorealism"]

        self.light = None
        self.camera = None
        self.camera_mtx = None

        self.mask_rendering = False

        self.reset()


    def reset(self):
        """
        Reset the state of the renderer:
        1. set Blender to its factory settings
        2. set center of mass for mesh objects
        3. create rendering node tree
        """

        # delete light, if any
        if self.light is not None:
            bpy.ops.object.select_all(action="DESELECT")
            self.light.select = True
            bpy.ops.object.delete()

        # delete light, if any
        if self.camera is not None:
            bpy.ops.object.select_all(action="DESELECT")
            self.camera.select = True
            bpy.ops.object.delete()

        self.light = None
        self.camera = None
        self.camera_mtx = None

        self._reset_blender()
        self._setup_scene()
        self._setup_renderer()

        self.camera, self.camera_mtx = self._create_camera(self.focal, self.z_min, self.z_max)


    def _reset_blender(self):
        """
        Initialize Blender with its factory settings.
        Remove all objects from the scene.
        """

        # restore factory settings
        #bpy.ops.wm.read_factory_settings()
        for scene in bpy.data.scenes:
            for obj in scene.objects:
                scene.objects.unlink(obj)

        # consider only the objects in the default scene
        data = [
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras
        ]
        for bpy_data_iter in data:
            for id_data in bpy_data_iter:
                bpy_data_iter.remove(id_data)


    def _setup_scene(self):
        """
        Set center of mass as default for all meshes.
        Delete all other objects.
        """

        scene = bpy.context.scene

        bpy.ops.object.select_all(action="DESELECT")

        # remove non mesh objects
        for obj in scene.objects:
            obj.select = (obj.type != "MESH")
        bpy.ops.object.delete()

        # empty sequences are false by default
        if scene.objects:

            # unlink objects (all meshes) from parents
            bpy.ops.object.select_all()
            bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

            # join all meshes in one single object
            scene.objects.active = bpy.data.objects[0]
            bpy.ops.object.join()
            bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            bpy.context.object.name = "Object"
            bpy.context.object.dimensions = bpy.context.object.dimensions / max(bpy.context.object.dimensions)

        # set the origin of the object to the cursor location
        scene.cursor_location = [0, 0, 0]
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        # bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN", center="BOUNDS")
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")

        if self.add_ground_plane:
            bpy.ops.mesh.primitive_plane_add(radius=10.)

        bpy.ops.object.select_all(action="DESELECT")


    def _setup_renderer(self):

        scene = bpy.context.scene

        # centimeters
        scene.unit_settings.system = "METRIC"

        # rendering settings
        scene.render.resolution_x = self.width
        scene.render.resolution_y = self.height
        scene.render.resolution_percentage = 100
        scene.render.use_antialiasing = False
        scene.render.use_file_extension = False

        scene.render.image_settings.file_format = self.outputext
        scene.render.image_settings.color_mode = self.colormode
        scene.render.image_settings.color_depth = self.colordepth
        scene.render.image_settings.compression = 100

        if self.light is None:
            self._create_light(mode="SUN")

        # background color
        scene.world.horizon_color = (0, 0, 0)

        # photorealism options
        if self.photorealism:
            # ambient occlusion
            scene.world.light_settings.use_ambient_occlusion = True
            scene.world.light_settings.ao_factor = 1
            scene.world.light_settings.ao_blend_type = "MULTIPLY"
            # environment light
            scene.world.light_settings.use_environment_light = True
            scene.world.light_settings.environment_energy = 0.2
            scene.world.light_settings.environment_color = "PLAIN"
            # indirect light (only works with approximate gather method)
            scene.world.light_settings.gather_method = "APPROXIMATE"
            scene.world.light_settings.use_indirect_light = True
            scene.world.light_settings.indirect_factor = 1
            # compute shadows from light source
            self.light.data.shadow_method = "RAY_SHADOW"
            self.light.data.shadow_ray_samples = 12
            self.light.data.shadow_soft_size = 10
            self.light.data.use_specular = True # see object material
            # enable anti-aliasing
            scene.render.use_antialiasing = True
            scene.render.antialiasing_samples = "8"
            scene.render.pixel_filter_type = "CATMULLROM"
            # add light to rendering process
            self.light.hide_render = False
            self.light.hide = False
        else:
            # remove light from rendering process
            self.light.hide_render = True
            self.light.hide = True

        # create the rendering node graph
        scene.use_nodes = True

        # delete all nodes but the render layer
        for node in scene.node_tree.nodes:
            node.select = (node.name != "Render Layers")
        for node in [node for node in scene.node_tree.nodes if node.select]:
            scene.node_tree.nodes.remove(node)

        # enable pass indexing
        scene.render.layers["RenderLayer"].use_pass_object_index = True

        # add the rgb output node
        rgb_output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
        rgb_output_node.name = "RGBOutNode"
        rgb_output_node.format.file_format = self.outputext
        rgb_output_node.format.color_mode = "RGB"
        rgb_output_node.format.color_depth = "8"

        # link the rgb render node to the output
        scene.node_tree.links.new(scene.node_tree.nodes["Render Layers"].outputs["Image"], rgb_output_node.inputs[0])

        # from blender 2.79 renderer_node.outputs['Z'] has become renderer_node.outputs['Depth']
        BLENDER_VERSION_MAJOR, BLENDER_VERSION_MINOR = [int(x) for x in bpy.app.version_string.split()[0].split(".")]
        depth_output_name = "Depth" if BLENDER_VERSION_MAJOR >= 2 and BLENDER_VERSION_MINOR >= 79 else "Z"

        # node to normalize z values
        normalizer_node = scene.node_tree.nodes.new("CompositorNodeMath")
        normalizer_node.name = "NormalizerNode"
        normalizer_node.operation = "DIVIDE"
        normalizer_node.inputs[1].default_value = self.z_max
        scene.node_tree.links.new(scene.node_tree.nodes["Render Layers"].outputs[depth_output_name], normalizer_node.inputs[0])

        depth_output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
        depth_output_node.name = "DepthOutNode"
        depth_output_node.format.file_format = self.outputext
        depth_output_node.format.color_mode = self.colormode
        depth_output_node.format.color_depth = self.colordepth

        # link the normalizer node to the output
        scene.node_tree.links.new(normalizer_node.outputs["Value"], depth_output_node.inputs[0])

        # set output paths and filenames
        self.mask_rendering = False
        self.set_destination_paths(self.outputfolder)

        # TODO
        # the save image need to be scaled with a factor: (Z_MAX / 2^COLOR_DEPTH)

        return
    def enable_mask_rendering(self):

        if self.mask_rendering:
            return True

        scene = bpy.context.scene
        objects = [ob for ob in bpy.data.objects if ob.pass_index > 0]

        nobjs = len(objects)
        if nobjs == 0:
            return False

        # each object will be drawn to the mask as a group of pixels
        # with the same value (i.e. its pass_index)
        # in order to save the mask, we scale the pass_index to the
        # range 0-255 (normalized to 0-1)
        # the maximum number of object is then 254 (0 is background)
        nodes = []
        for obj in objects:
            # do not consider background objects
            if obj.pass_index == 0:
                continue
            # the object is rendered alone (according to its id)
            idmask_node = scene.node_tree.nodes.new("CompositorNodeIDMask")
            idmask_node.index = obj.pass_index
            scene.node_tree.links.new(scene.node_tree.nodes["Render Layers"].outputs["IndexOB"], idmask_node.inputs[0])
            # scale the value to 1/255 since blender works between 0 and 1
            scalevalue_node = scene.node_tree.nodes.new("CompositorNodeMath")
            scalevalue_node.operation = "MULTIPLY"
            scalevalue_node.inputs[1].default_value = obj.pass_index / 255.
            scene.node_tree.links.new(idmask_node.outputs["Alpha"], scalevalue_node.inputs[0])
            # save a reference to the object processing node
            nodes.append(scalevalue_node)
            # save the object mask as a single image
            objmask_output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
            objmask_output_node.name = "ObjMaskOutNode" + str(obj.pass_index)
            objmask_output_node.format.file_format = self.outputext
            objmask_output_node.format.color_mode = "RGB"
            objmask_output_node.format.color_depth = "8"
            scene.node_tree.links.new(idmask_node.outputs["Alpha"], objmask_output_node.inputs[0])


        # the first object-tree (of nodes) is considered as reference
        output_node = nodes[0]

        # for each objects in addition to the first object (if any) is added
        # to the reference node, that is updated at each iteration
        if nobjs > 1:
            for i in range(1, nobjs):
                add_node = scene.node_tree.nodes.new("CompositorNodeMath")
                add_node.operation = "ADD"
                scene.node_tree.links.new(output_node.outputs["Value"], add_node.inputs[0])
                scene.node_tree.links.new(nodes[i].outputs["Value"], add_node.inputs[1])
                output_node = add_node

        # save the combined image
        mask_output_node = scene.node_tree.nodes.new("CompositorNodeOutputFile")
        mask_output_node.name = "MaskOutNode"
        mask_output_node.format.file_format = self.outputext
        mask_output_node.format.color_mode = "RGB"
        mask_output_node.format.color_depth = "8"

        # link the normalizer node to the output
        scene.node_tree.links.new(output_node.outputs["Value"], mask_output_node.inputs[0])

        # enable mask
        self.mask_rendering = True
        return True


    def set_destination_paths(self, folder, subfolder="", rgb_name=None, depth_name=None, mask_name=None):

        # set the global output path (only to avoid "/tmp/: is a directory" on stderr)
        bpy.context.scene.render.filepath = "/tmp/fart.png"

        # update path names
        self.outputfolder = folder

        scene = bpy.context.scene
        # set output paths and filenames
        if rgb_name is not None and isinstance(rgb_name, str):
            rgb_output_node = scene.node_tree.nodes["RGBOutNode"]
            scene.node_tree.links.new(scene.node_tree.nodes["Render Layers"].outputs["Image"], rgb_output_node.inputs[0])
            rgb_output_node.base_path = os.path.join(self.outputfolder, "rgb", subfolder)
            rgb_output_node.file_slots[0].path = "####_" + rgb_name + "." + self.outputext.lower()
        else: # disable rendering
            links = scene.node_tree.nodes["Render Layers"].outputs["Image"].links
            if len(links) > 0:
                scene.node_tree.links.remove(links[0])

        BLENDER_VERSION_MAJOR, BLENDER_VERSION_MINOR = [int(x) for x in bpy.app.version_string.split()[0].split(".")]
        depth_output_name = "Depth" if BLENDER_VERSION_MAJOR >= 2 and BLENDER_VERSION_MINOR >= 79 else "Z"
        if depth_name is not None and isinstance(depth_name, str):
            normalizer_node = scene.node_tree.nodes["NormalizerNode"]
            scene.node_tree.links.new(scene.node_tree.nodes["Render Layers"].outputs[depth_output_name], normalizer_node.inputs[0])
            depth_output_node = scene.node_tree.nodes["DepthOutNode"]
            depth_output_node.base_path = os.path.join(self.outputfolder, "depth", subfolder)
            depth_output_node.file_slots[0].path = "####_" + depth_name + "." + self.outputext.lower()
        else:
            links = scene.node_tree.nodes["Render Layers"].outputs[depth_output_name].links
            if len(links) > 0:
                scene.node_tree.links.remove(links[0])

        if mask_name is not None and self.mask_rendering:
            # labels mask
            scene.node_tree.nodes["MaskOutNode"].base_path = os.path.join(self.outputfolder, "mask", subfolder)
            scene.node_tree.nodes["MaskOutNode"].file_slots[0].path = "####_" + mask_name + "." + self.outputext.lower()
            # single object masks
            objects = [ob for ob in bpy.data.objects if ob.pass_index > 0]
            for obj in objects:
                suffix = str(obj.pass_index)
                scene.node_tree.nodes["ObjMaskOutNode" + suffix].base_path = os.path.join(self.outputfolder, "objmask", mask_name, subfolder)
                scene.node_tree.nodes["ObjMaskOutNode" + suffix].file_slots[0].path = "####_" + suffix + "." + self.outputext.lower()


    def _update_camera_mtx(self):
        scene = bpy.context.scene
        scale = scene.render.resolution_percentage / 100
        width = scene.render.resolution_x * scale # px
        height = scene.render.resolution_y * scale # px
        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(self.camera.data.angle / 2)
        K[1][1] = height / 2. / np.tan(self.camera.data.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        self.camera_mtx = K


    def _create_camera(self, focal, z_min, z_max):
        cam = bpy.data.cameras.new("Camera")
        cam.lens = focal
        cam.clip_start = z_min
        cam.clip_end = z_max
        camera = bpy.data.objects.new("Camera", cam)
        bpy.context.scene.objects.link(camera)
        bpy.context.scene.camera = camera
        self.camera = camera
        self._update_camera_mtx()
        return self.camera, self.camera_mtx


    def set_camera_focal(self, focal):
        self.camera.data.lens = focal
        self._update_camera_mtx()


    def set_camera_clips(self, z_min, z_max):
        self.camera.data.clip_start = z_min
        self.camera.data.clip_end = z_max


    def set_camera_pose(self, world_pose_mtx):
        self.enable_tracking(self.camera, enable=False)
        self.camera.matrix_world = world_pose_mtx


    def set_camera_location(self, location):
        self.camera.location = location


    def _create_light(self, mode="SUN"):
        # create light source entity
        light = bpy.data.lamps.new("Light", type=mode)
        # create light object and link to scene
        light = bpy.data.objects.new("Light", light)
        bpy.context.scene.objects.link(light)
        self.light = light
        return light


    def set_light_location(self, location):
        self.light.location = location


    def enable_tracking(self, entity, enable=True):
        # create tracking constraint
        if not entity.constraints:
            if enable:
                tracker = entity.constraints.new(type="TRACK_TO")
                tracker.up_axis = "UP_Y"
                tracker.track_axis = "TRACK_NEGATIVE_Z"
            else:
                return
        # enable/disable tracking
        entity.constraints["Track To"].mute = not enable


    def set_target(self, entity, target):
        self.enable_tracking(entity)
        if isinstance(target, list) or isinstance(target, tuple) or isinstance(target, np.ndarray):
            loc = target[0], target[1], target[2]
            bpy.ops.object.empty_add(type='CUBE', radius=1, location=loc)
            target = bpy.data.objects["Empty"]
        entity.constraints["Track To"].target = target


    def import_object(self, filename, pose=np.eye(4), size=None, oid=1):
        """
        Import a model file into blender engine.
        It is possible to set its pose and set the new maximum
        size of the model.

        Parameters
        ----------
        filename : string
            Path off the model file. Currently supported formats: PLY
        pose : 4x4 numpy.ndarray (default: numpy.eye(4))
            Pose of the object to set right after loading.
            The 4x4 matrix is in the format [R t; 0 0 0 1] where R is
            a 3x3 rotation matrix and t the 3x1 location of the object.
        size : float (default: None)
            The new size of the object. If not None, model dimensions
            are normalized by the maximum dimension and scaled by size.

        Returns
        -------
        model : bpy.data.objects item
            Reference to the model as blender object
        """

        # extract name and extension of the model file
        name, ext = os.path.basename(filename).split(".")

        # load model according to file extension
        if ext == "ply":
            bpy.ops.import_mesh.ply(filepath=filename)
        else:
            raise NotImplementedError()

        # the name of the file is assigned
        # to the mesh object in blender engine
        model = bpy.data.objects[name]
        model.name = name + str(oid)

        # set object reference point (origin) and pose
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="BOUNDS")
        self.set_model_pose(model, pose)

        # normalize and scale model dimensions
        if size is not None:
            model.dimensions = size * model.dimensions / max(model.dimensions)

        # add material
        # FIXME: adjust properties
        material = bpy.data.materials.new(name="Material")
        material.specular_intensity = 0.25
        model.data.materials.append(material)
        # enable vertex color rendering
        # this is necessary to render the vertex color
        # in the rgb branch of the rendering node tree
        model.active_material.use_vertex_color_paint = True

        # if rendering is not photorealistic, render only
        # the vertex color information of the model
        if not self.photorealism:
            model.active_material.use_shadeless = True

        # set object id
        model.pass_index = oid

        return model


    def set_model_pose(self, model, pose):
        R = pose[:3,:3]
        t = pose[:3,-1]
        self.set_model_rotation(model, R)
        self.set_model_location(model, t)


    def set_model_rotation(self, model, R):
        model.rotation_euler = rotation_to_euler(R)


    def set_model_location(self, model, t):
        model.location = t


    def check_intersections(self, objA, objB):

        def bounding_box(obj):
            # get rotation matrix
            theta1, theta2, theta3 = np.asarray(obj.rotation_euler)
            R = euler_to_rotation(theta1, theta2, theta3)
            # extract vertices
            verts = np.asarray([np.asarray(v.co) for v in obj.data.vertices])
            # apply transformations
            verts = np.matmul(R, verts.T).T
            verts = verts * obj.scale
            verts = verts + obj.location
            # compute boundaries
            x0,y0,z0 = np.min(verts, axis=0)
            x1,y1,z1 = np.max(verts, axis=0)
            return [x0,x1,y0,y1,z0,z1]

        bboxA = bounding_box(objA)
        bboxB = bounding_box(objB)

        x_check = bboxA[0] <= bboxB[1] and bboxA[1] >= bboxB[0]
        y_check = bboxA[2] <= bboxB[3] and bboxA[3] >= bboxB[2]
        z_check = bboxA[4] <= bboxB[5] and bboxA[5] >= bboxB[4]

        return x_check and y_check and z_check


    def _render(self):
        try:
            bpy.ops.render.render(write_still=True, use_viewport=True)
        except:
            pass

    def render(self):

        self._render()

        if self.mask_rendering:
            scene = bpy.context.scene

            # prepare paths
            rgb_out_base_path = scene.node_tree.nodes["RGBOutNode"].base_path
            rgb_out_fpath = scene.node_tree.nodes["RGBOutNode"].file_slots[0].path
            depth_out_base_path = scene.node_tree.nodes["DepthOutNode"].base_path
            depth_out_fpath = scene.node_tree.nodes["DepthOutNode"].file_slots[0].path
            mask_out_base_path = scene.node_tree.nodes["MaskOutNode"].base_path
            mask_out_fpath = scene.node_tree.nodes["MaskOutNode"].file_slots[0].path
            scene.node_tree.nodes["RGBOutNode"].base_path = "/tmp/fart"
            scene.node_tree.nodes["RGBOutNode"].file_slots[0].path = "rgb.png"
            scene.node_tree.nodes["DepthOutNode"].base_path = "/tmp/fart"
            scene.node_tree.nodes["DepthOutNode"].file_slots[0].path = "depth.png"
            scene.node_tree.nodes["MaskOutNode"].base_path = "/tmp/fart"
            scene.node_tree.nodes["MaskOutNode"].file_slots[0].path = "mask.png"

            # disable all objects
            objects = [ob for ob in bpy.data.objects if ob.pass_index > 0]
            paths = []
            for obj in objects:
                obj.hide_render = True
                bp = scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].base_path
                fn = scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].file_slots[0].path
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].base_path = "/tmp/fart"
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].file_slots[0].path = "objmask.png"
                paths.append([bp, fn])
            # render the scene with only the target object
            for i,obj in enumerate(objects):
                bp, fn = paths[i]
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].base_path = bp
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].file_slots[0].path = fn
                obj.hide_render = False
                self._render()
                obj.hide_render = True
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].base_path = "/tmp/fart"
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].file_slots[0].path = "objmask.png"
            # enable all objects
            for i,obj in enumerate(objects):
                obj.hide_render = False
                bp, fn = paths[i]
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].base_path = bp
                scene.node_tree.nodes["ObjMaskOutNode" + str(obj.pass_index)].file_slots[0].path = fn

            scene.node_tree.nodes["RGBOutNode"].base_path = rgb_out_base_path
            scene.node_tree.nodes["RGBOutNode"].file_slots[0].path = rgb_out_fpath
            scene.node_tree.nodes["DepthOutNode"].base_path = depth_out_base_path
            scene.node_tree.nodes["DepthOutNode"].file_slots[0].path = depth_out_fpath
            scene.node_tree.nodes["MaskOutNode"].base_path = mask_out_base_path
            scene.node_tree.nodes["MaskOutNode"].file_slots[0].path = mask_out_fpath

        # rename output imaget to remove blender default prefix (0001_)
        filelist = glob.glob(os.path.join(self.outputfolder, "**", "0001_*." + self.outputext.lower()), recursive=True)
        for fname in filelist:
            dst = os.path.join( os.path.dirname(fname), os.path.basename(fname)[5:] )
            os.rename(fname, dst)
