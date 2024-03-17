import os
import sys
import torch
from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.runtime import Core, Type, PartialShape, serialize
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._dynamo.backends.registry import register_backend
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten
from types import MappingProxyType
from hashlib import sha256
import functools
from modules import shared, devices

def BUILD_MAP_UNPACK(self, inst):
        items = self.popn(inst.argval)
        # ensure everything is a dict
        items = [BuiltinVariable(dict).call_function(self, [x], {}) for x in items] # noqa: F821
        result = dict()
        for x in items:
            assert isinstance(x, ConstDictVariable) # noqa: F821
        result.update(x.items)
        self.push(
            ConstDictVariable( # noqa: F821
                result,
                dict,
                mutable_local=MutableLocal(), # noqa: F821
                **VariableTracker.propagate(items), # noqa: F821
            )
        )
tmp_torch = sys.modules["torch"]
tmp_torch.BUILD_MAP_UNPACK_WITH_CALL = BUILD_MAP_UNPACK

compiled_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}

DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, file_name=""):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.file_name = file_name
        self.perm_fallback = False

    def __call__(self, *args):
        #if self.perm_fallback:
        #    return self.gm(*args)

        #try:
        result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, file_name=self.file_name)
        #except Exception:
        #    self.perm_fallback = True
        #    return self.gm(*args)

        return result

def get_device():
    core = Core()
    if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
    elif shared.opts.openvino_hetero_gpu:
        device = ""
        available_devices = core.available_devices
        available_devices.remove("CPU")
        if shared.opts.openvino_remove_igpu_from_hetero and "GPU.0" in available_devices:
            available_devices.remove("GPU.0")
        for gpu in available_devices:
            device = f"{device},{gpu}"
        if not shared.opts.openvino_remove_cpu_from_hetero:
            device = f"{device},CPU"
        device = f"HETERO:{device[1:]}"
    elif any(openvino_cpu in cpu_module.lower() for cpu_module in shared.cmd_opts.use_cpu for openvino_cpu in ["openvino", "all"]):
        device = "CPU"
    elif shared.cmd_opts.device_id is not None:
        device = f"GPU.{shared.cmd_opts.device_id}"
    elif "GPU" in core.available_devices:
        device = "GPU"
    elif "GPU.1" in core.available_devices:
        device = "GPU.1"
    elif "GPU.0" in core.available_devices:
        device = "GPU.0"
    else:
        device = "CPU"
        shared.log.warning("OpenVINO: No compatible GPU detected!")
    os.environ.setdefault('OPENVINO_TORCH_BACKEND_DEVICE', device)
    return device

def get_openvino_device():
    core = Core()
    try:
        return core.get_property(get_device(), "FULL_DEVICE_NAME")
    except Exception:
        return f"OpenVINO {get_device()}"

def cache_root_path():
    cache_root = "./cache/"
    if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
        cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")
    return cache_root

def cached_model_name(model_hash_str, device, args, cache_root, reversed = False):
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        shared.log.error(f"Cache directory {cache_root} cannot be created. Model caching is disabled. Error: {error}")
        return None

    inputs_str = ""
    for input_data in args:
        if reversed:
            inputs_str = "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "") + inputs_str
        else:
            inputs_str += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
    inputs_str = sha256(inputs_str.encode('utf-8')).hexdigest()
    file_name += inputs_str

    return file_name

def check_fully_supported(self, graph_module: GraphModule) -> bool:
    num_fused = 0
    for node in graph_module.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            num_fused += 1
        elif node.op != "placeholder" and node.op != "output":
            return False
    if num_fused == 1:
        return True
    return False

Partitioner.check_fully_supported = functools.partial(check_fully_supported, Partitioner)

def execute(
    gm,
    *args,
    executor = "openvino",
    executor_parameters = None,
    file_name = ""
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, file_name=file_name)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters, file_name=file_name)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)

def execute_cached(compiled_model, *args):
    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    if (shared.compiled_model_state.cn_model == []):
        ov_inputs.reverse()

    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result

def openvino_clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()

def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None, file_name=""):
    core = Core()

    device = get_device()
    cache_root = cache_root_path()

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for input_data in args:
            input_types.append(input_data.type())
            input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)

        im = fe.load(decoder)

        om = fe.convert(im)

        if (file_name is not None):
            serialize(om, file_name + ".xml", file_name + ".bin")
            if (shared.compiled_model_state.cn_model != []):
                f = open(file_name + ".txt", "w")
                for input_data in args:
                    f.write(str(input_data.size()))
                    f.write("\n")
                f.close()

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(args):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    if model_hash_str is not None:
        core.set_property({'CACHE_DIR': cache_root + '/blob'})

    compiled = core.compile_model(om, device)
    return compiled

def openvino_compile_cached_model(cached_model_path, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(example_inputs):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    core.set_property({'CACHE_DIR': cache_root_path() + '/blob'})

    compiled_model = core.compile_model(om, get_device())

    return compiled_model

def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        model_hash_str = model_hash_str + str(partition_id)

    if use_cache and (partition_id in compiled_cache):
        compiled = compiled_cache[partition_id]
    else:
        if (shared.compiled_model_state.cn_model != [] and file_name is not None
                and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin")):
            compiled = openvino_compile_cached_model(file_name, *args)
        else:
            compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, file_name=file_name)
        compiled_cache[partition_id] = compiled

    flat_args, _ = tree_flatten(args)
    ov_inputs = [a.detach().cpu().numpy() for a in flat_args]

    res = compiled(ov_inputs)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1

def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, file_name=""):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    for idx, input_data in enumerate(args):
        if isinstance(input_data, torch.Tensor):
            signature = signature + "_" + str(idx) + ":" + str(input_data.type())[6:] + ":" + str(input_data.size())[11:-1].replace(" ", "")
        else:
            signature = signature + "_" + str(idx) + ":" + type(input_data).__name__ + ":val(" + str(input_data) + ")"

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache,
                                                         model_hash_str=model_hash_str, file_name=file_name)

    return partitioned_modules[signature](*args)

def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, file_name=""):
    global max_openvino_partitions
    for node in gm.graph.nodes:
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, shared.compiled_model_state.partition_id, use_python_fusion_cache,
                        model_hash_str=model_hash_str, file_name=file_name),
            )
            shared.compiled_model_state.partition_id = shared.compiled_model_state.partition_id + 1

    return gm

@register_backend
@fake_tensor_unsupported
def openvino_fx(subgraph, example_inputs):
    executor_parameters = None
    inputs_reversed = False
    if not shared.opts.openvino_disable_model_caching:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "1")
        # Create a hash to be used for caching
        model_hash_str = sha256(subgraph.code.encode('utf-8')).hexdigest()
        if (shared.compiled_model_state.cn_model != [] and shared.compiled_model_state.partition_id == 0):
            model_hash_str = model_hash_str + str(shared.compiled_model_state.cn_model)

        if (shared.compiled_model_state.lora_model != []):
            model_hash_str = model_hash_str + str(shared.compiled_model_state.lora_model)

        executor_parameters = {"model_hash_str": model_hash_str}
        # Check if the model was fully supported and already cached
        example_inputs.reverse()
        inputs_reversed = True
        maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", get_device(), example_inputs, cache_root_path())

        if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
            if (shared.compiled_model_state.cn_model != [] and str(shared.compiled_model_state.cn_model) in maybe_fs_cached_name):
                example_inputs_reordered = []
                if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                    f = open(maybe_fs_cached_name + ".txt", "r")
                    for input_data in example_inputs:
                        shape = f.readline()
                        if (str(input_data.size()) != shape):
                            for idx1, input_data1 in enumerate(example_inputs):
                                if (str(input_data1.size()).strip() == str(shape).strip()):
                                    example_inputs_reordered.append(example_inputs[idx1])
                    example_inputs = example_inputs_reordered

                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, *example_inputs)

                def _call(*args):
                    if (shared.compiled_model_state.cn_model != [] and str(shared.compiled_model_state.cn_model) in maybe_fs_cached_name):
                        args_reordered = []
                        if (os.path.isfile(maybe_fs_cached_name + ".txt")):
                            f = open(maybe_fs_cached_name + ".txt", "r")
                            for input_data in args:
                                shape = f.readline()
                                if (str(input_data.size()) != shape):
                                    for idx1, input_data1 in enumerate(args):
                                        if (str(input_data1.size()).strip() == str(shape).strip()):
                                            args_reordered.append(args[idx1])
                        args = args_reordered

                    res = execute_cached(compiled_model, *args)
                    shared.compiled_model_state.partition_id = shared.compiled_model_state.partition_id + 1
                    return res
                return _call
    else:
        os.environ.setdefault('OPENVINO_TORCH_MODEL_CACHING', "0")
        maybe_fs_cached_name = None

    if inputs_reversed:
        example_inputs.reverse()
    model = make_fx(subgraph)(*example_inputs)
    for node in model.graph.nodes:
        if node.target == torch.ops.aten.mul_.Tensor:
            node.target = torch.ops.aten.mul.Tensor
    with devices.inference_context():
        model.eval()
    partitioner = Partitioner()
    compiled_model = partitioner.make_partitions(model)

    if executor_parameters is not None and 'model_hash_str' in executor_parameters:
        # Check if the model is fully supported.
        fully_supported = partitioner.check_fully_supported(compiled_model)
        if fully_supported:
            executor_parameters["model_hash_str"] += "_fs"

    def _call(*args):
        res = execute(compiled_model, *args, executor="openvino",
                        executor_parameters=executor_parameters, file_name=maybe_fs_cached_name)
        return res
    return _call
