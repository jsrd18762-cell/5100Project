"""Step8 深层 float32 分块执行：GPU计算，CPU权重，磁盘梯度/Adam状态。

不切分样本batch，不改变BN/Dropout/网络/损失/Adam公式。Linear逐层完整
矩阵计算，仅梯度传输和Adam状态按参数元素分块，限制同时驻留显存的量。
项目路径相对trial；本文件对应notebooks/streamed_training.ipynb。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
from torch import nn
from torch.optim.adam import adam as torch_adam

BACKEND = 'float32_disk_stream_adam_v1'
CHUNK_BYTES = 64 * 2**20
CHUNK_ELEMENTS = CHUNK_BYTES // 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ensure_folder(folder):
    path = Path(folder)
    require(not path.is_absolute() and path.parts[0] == 'step8' and '..' not in path.parts,
            '分块执行的工作文件必须在step8内。')
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunks(total, size=CHUNK_ELEMENTS):
    for start in range(0, total, size):
        yield start, min(total, start + size)


def checkpoint_metadata(path):
    """仅读取标准torch.save的结构和字节偏移，不整体映射巨型checkpoint。

    Windows的整文件私有mmap仍可能耗尽系统提交额度。FakeTensorMode只读取
    张量元数据，真实权重/moment随后以最多64MiB的只读窗口逐片复制。
    _checkpoint_offset由当前锁定的PyTorch提供；不存在时明确停止。
    """
    from torch._subclasses.fake_tensor import FakeTensorMode
    with zipfile.ZipFile(path) as archive:
        byteorders = [name for name in archive.namelist() if name.endswith('/byteorder')]
        require(len(byteorders) == 1 and archive.read(byteorders[0]) == b'little',
                'checkpoint必须使用当前环境的小端torch.save格式。')
    with FakeTensorMode():
        return torch.load(path, map_location='cpu', weights_only=True)


def checkpoint_slice(path, tensor, start, end):
    """返回独立CPU Tensor；退出后不保留checkpoint的映射或文件句柄。"""
    dtypes = {torch.float32: '<f4', torch.int64: '<i8', torch.uint8: 'u1'}
    require(tensor.dtype in dtypes and tensor.is_contiguous(),
            f'不支持的checkpoint张量类型/布局：{tensor.dtype}/{tensor.shape}')
    require(0 <= start < end <= tensor.numel(), 'checkpoint读取范围错误。')
    offset = getattr(tensor.untyped_storage(), '_checkpoint_offset', None)
    require(isinstance(offset, int) and offset >= 0, '当前PyTorch未提供checkpoint字节偏移。')
    offset += (tensor.storage_offset()+start)*tensor.element_size()
    require(offset+(end-start)*tensor.element_size() <= Path(path).stat().st_size,
            'checkpoint张量超出文件边界。')
    mapped = np.memmap(path, dtype=dtypes[tensor.dtype], mode='r',
                       offset=offset, shape=(end-start,))
    try:
        return torch.from_numpy(np.array(mapped, copy=True))
    finally:
        mapped._mmap.close()


@torch.no_grad()
def load_model_checkpoint(model, path, state):
    """严格核验state_dict名称/形状/精度，向预分配CPU/CUDA参数逐片复制。"""
    targets = model.state_dict()
    require(targets.keys() == state.keys(), 'checkpoint参数/BN名称不一致。')
    for name, target in targets.items():
        source = state[name]
        require(source.shape == target.shape and source.dtype == target.dtype and target.is_contiguous(),
                f'checkpoint形状/精度不一致：{name}')
        for start, end in chunks(source.numel(), CHUNK_BYTES//source.element_size()):
            value = checkpoint_slice(path, source, start, end)
            require(bool(torch.isfinite(value).all()), f'checkpoint含非有限值：{name}')
            target.reshape(-1)[start:end].copy_(value, non_blocking=False)


def materialize_metadata(path, value):
    """只还原小型游标/排列/RNG张量；禁止把模型或Adam状态交给本函数。"""
    if torch.is_tensor(value):
        require(value.numel()*value.element_size() <= CHUNK_BYTES, '元数据张量异常过大。')
        if value.numel() == 0:
            return torch.empty(tuple(value.shape), dtype=value.dtype)
        return checkpoint_slice(path, value, 0, value.numel()).reshape(value.shape)
    if isinstance(value, dict):
        return {key: materialize_metadata(path, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(materialize_metadata(path, item) for item in value)
    if isinstance(value, list):
        return [materialize_metadata(path, item) for item in value]
    return value


class DiskState:
    """工作缓存不是恢复依据；只有完整原子checkpoint可用于恢复。

    每次只映射一个64MiB片段，读写完成立即关闭映射，防止整个磁盘状态
    被长期锁在主内存。新建moment文件在第一次更新时直接写入零状态的
    更新结果，不读取未初始化区域。所有文件均为原生小端float32。
    """
    def __init__(self, folder, named_parameters):
        self.folder = ensure_folder(folder)
        self.parameters = dict(named_parameters)
        self.names = list(self.parameters)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.ready = set()
        for name, parameter in self.parameters.items():
            require(parameter.dtype == torch.float32 and parameter.is_contiguous(),
                    f'参数必须为连续float32：{name}')
            for kind in ('gradient', 'mean', 'variance'):
                # 只截断本后端明确拥有的缓存文件；断点本身不在该目录。
                with self.path(name, kind).open('wb') as stream:
                    stream.truncate(parameter.numel() * 4)

    def path(self, name, kind):
        return self.folder / f'{self.index[name]:03d}_{kind}.f32.bin'

    @contextlib.contextmanager
    def mapped(self, name, kind, start, end, mode='r+'):
        require(0 <= start < end <= self.parameters[name].numel(), '磁盘片段范围错误。')
        array = np.memmap(self.path(name, kind), dtype='<f4', mode=mode,
                          offset=start * 4, shape=(end-start,))
        try:
            yield array
        finally:
            # CPU/GPU复制必须在退出前完成，不返回引用映射的Tensor。
            array._mmap.close()

    def read_gpu(self, name, kind, start, end, device):
        with self.mapped(name, kind, start, end) as array:
            value = torch.from_numpy(array).to(device, non_blocking=False)
        return value

    def write(self, name, kind, start, value):
        cpu = value.detach().reshape(-1).to('cpu', non_blocking=False)
        with self.mapped(name, kind, start, start+cpu.numel()) as array:
            array[:] = cpu.numpy()

    def write_gradient(self, name, start, value):
        require(bool(torch.isfinite(value).all()), f'非有限梯度：{name}')
        self.write(name, 'gradient', start, value)

    def finish_gradient(self, name):
        require(name not in self.ready, f'本网络不允许参数共享/重复反传：{name}')
        self.ready.add(name)

    def remove_work_files(self):
        """仅在最终权重保存成功后清理本模型拥有的三类临时缓存。"""
        for name in self.names:
            for kind in ('gradient','mean','variance'):
                self.path(name,kind).unlink(missing_ok=True)
        if not any(self.folder.iterdir()):
            self.folder.rmdir()


class _StreamedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, disk, name, chunk_elements):
        require(inputs.ndim == 2 and inputs.dtype == torch.float32, 'Linear需要二维float32批次。')
        require(weight.device.type == 'cpu', '分块Linear权重必须保存在CPU。')
        # 最大单层约3.38GiB，当前8GiB显卡可放下。保持完整GEMM，不改变
        # 点积求和的划分，也不把BN/Dropout拆成小batch。
        w = weight.to(inputs.device, non_blocking=False)
        b = bias.to(inputs.device, non_blocking=False) if bias is not None else None
        result = nn.functional.linear(inputs, w, b)
        del w, b
        if weight.numel()*weight.element_size() >= 128*2**20:
            # 大层释放后归还空闲CUDA缓存，避免后续小分配把巨型段切碎。
            torch.cuda.empty_cache()
        ctx.save_for_backward(inputs, weight)
        ctx.disk, ctx.name, ctx.chunk_elements = disk, name, chunk_elements
        ctx.has_bias = bias is not None
        return result

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        require(ctx.disk is not None, '推理专用分块模型不允许反向传播。')
        disk, name = ctx.disk, ctx.name
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            w = weight.to(inputs.device, non_blocking=False)
            grad_inputs = grad_output @ w
            # 先释放完整GPU权重，再创建完整权重梯度，二者不同时占显存。
            del w
            if weight.numel()*weight.element_size() >= 128*2**20:
                torch.cuda.empty_cache()
        grad_weight = grad_output.T @ inputs
        for start, end in chunks(weight.numel(), ctx.chunk_elements):
            disk.write_gradient(name+'.weight', start, grad_weight.reshape(-1)[start:end])
        del grad_weight
        if weight.numel()*weight.element_size() >= 128*2**20:
            torch.cuda.empty_cache()
        if ctx.has_bias:
            disk.write_gradient(name+'.bias', 0, grad_output.sum(dim=0))
        disk.finish_gradient(name+'.weight')
        if ctx.has_bias:
            disk.finish_gradient(name+'.bias')
        return grad_inputs, None, None, None, None, None


class StreamedLinear(nn.Linear):
    """保留Linear的参数名称/形状/state_dict，不重新初始化已建立的权重。"""
    def __init__(self, original, disk, name, device, chunk_elements=CHUNK_ELEMENTS):
        nn.Module.__init__(self)
        require(original.weight.device.type == 'cpu', '安装前Linear应位于CPU。')
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.disk, self.parameter_name = disk, name
        self.compute_device = torch.device(device)
        self.chunk_elements = chunk_elements

    def forward(self, inputs):
        require(inputs.device == self.compute_device, '分块Linear输入的计算设备不一致。')
        return _StreamedLinear.apply(inputs, self.weight, self.bias, self.disk,
                                     self.parameter_name, self.chunk_elements)


def install_streaming(model, device, disk=None, chunk_elements=CHUNK_ELEMENTS):
    """仅替换执行方式；BN保留完整batch并且每层只前向一次。"""
    device = torch.device(device)
    require(device.type == 'cuda', '该内存后端用于CUDA计算。')
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._modules[child_name] = StreamedLinear(module, disk, name, device, chunk_elements)
        elif isinstance(module, nn.BatchNorm1d):
            module.to(device)
    return model


class StreamedAdam:
    """使用当前PyTorch的单张量Adam实现，moment和梯度按块往返磁盘。

    所有参数在完整反传后才更新，每个参数每个batch只有一个共同step。
    不使用8-bit状态、量化、AMP、低秩近似、CPU Adam或分块batch。
    """
    streamed = True

    def __init__(self, model, disk, device, lr=.0002, chunk_elements=CHUNK_ELEMENTS):
        self.model, self.disk = model, disk
        self.parameters = dict(model.named_parameters())
        require(list(self.parameters) == disk.names, '优化器的参数顺序发生改变。')
        self.device = torch.device(device)
        self.lr, self.chunk_elements, self.steps = lr, chunk_elements, 0

    def zero_grad(self, set_to_none=True):
        require(set_to_none, '本后端使用set_to_none=True。')
        self.disk.ready.clear()
        for parameter in self.parameters.values():
            parameter.grad = None

    @torch.no_grad()
    def step(self):
        # BN等小参数由普通autograd产生梯度，先核验所有梯度再修改任何权重。
        for name, parameter in self.parameters.items():
            if name not in self.disk.ready:
                require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()),
                        f'缺少有效梯度：{name}')
                self.disk.write_gradient(name, 0, parameter.grad)
                self.disk.finish_gradient(name)
        require(self.disk.ready == set(self.parameters), '反传没有覆盖全部训练参数。')
        for name, parameter in self.parameters.items():
            flat = parameter.detach().reshape(-1)
            for start, end in chunks(parameter.numel(), self.chunk_elements):
                p = flat[start:end].to(self.device, non_blocking=False).clone()
                g = self.disk.read_gpu(name, 'gradient', start, end, self.device)
                if self.steps == 0:
                    mean, variance = torch.zeros_like(p), torch.zeros_like(p)
                else:
                    mean = self.disk.read_gpu(name, 'mean', start, end, self.device)
                    variance = self.disk.read_gpu(name, 'variance', start, end, self.device)
                # 每一块从同一旧step开始，不能让分块数改变Adam偏差校正。
                step = torch.tensor(float(self.steps), dtype=torch.float32, device='cpu')
                torch_adam([p], [g], [mean], [variance], [], [step],
                    foreach=False, capturable=False, differentiable=False, fused=False,
                    amsgrad=False, beta1=.9, beta2=.999, lr=self.lr,
                    weight_decay=0., eps=1e-8, maximize=False)
                require(bool(torch.isfinite(p).all()) and bool(torch.isfinite(variance).all()),
                        f'Adam产生非有限值：{name}')
                flat[start:end].copy_(p, non_blocking=False)
                self.disk.write(name, 'mean', start, mean)
                self.disk.write(name, 'variance', start, variance)
                del p, g, mean, variance, step
        self.steps += 1

    @contextlib.contextmanager
    def checkpoint_state(self):
        """零拷贝映射moment供torch.save顺序读取；保存后关闭全部映射。"""
        mappings, states = [], {}
        try:
            if self.steps:
                for name, parameter in self.parameters.items():
                    states[name] = {}
                    for kind in ('mean', 'variance'):
                        array = np.memmap(self.disk.path(name, kind), dtype='<f4', mode='r+',
                                          shape=tuple(parameter.shape))
                        mappings.append(array)
                        states[name][kind] = torch.from_numpy(array)
            value = dict(backend=BACKEND, steps=self.steps, lr=self.lr,
                         chunk_elements=self.chunk_elements, parameter_names=list(self.parameters), states=states)
            yield value
        finally:
            states.clear()
            for array in mappings:
                array._mmap.close()

    def load_state_dict(self, saved, checkpoint):
        """从metadata所指向的标准.pt中逐片恢复moment，避免整文件私有mmap。"""
        require(saved['backend'] == BACKEND and saved['parameter_names'] == list(self.parameters) and
                saved['lr'] == self.lr and saved['chunk_elements'] == self.chunk_elements,
                '分块优化器配置不一致，不能接续。')
        self.steps = int(saved['steps'])
        require(self.steps >= 0, 'Adam step无效。')
        if not self.steps:
            return
        for name, parameter in self.parameters.items():
            for kind in ('mean', 'variance'):
                incoming = saved['states'][name][kind]
                require(incoming.dtype == torch.float32 and incoming.shape == parameter.shape,
                        'Adam moment的形状或精度不符。')
                for start, end in chunks(parameter.numel(), self.chunk_elements):
                    value = checkpoint_slice(checkpoint, incoming, start, end)
                    require(bool(torch.isfinite(value).all()), 'Adam moment含非有限值。')
                    self.disk.write(name, kind, start, value)


def preflight(model_parameter_count, device, training=True, largest_linear_elements=907470000):
    """分块后显存与完整参数数目解耦；CPU仍需存放全部float32权重。"""
    import psutil
    free, _ = torch.cuda.mem_get_info(device)
    gpu_estimate = 1.5 * 2**30 + max(largest_linear_elements*4, 8*CHUNK_BYTES)
    # 调用前尚未创建巨型CPU模型；表达矩阵已加载则其占用已反映在available。
    host_estimate = model_parameter_count*4*1.15 + 1.5*2**30
    require(free > gpu_estimate, f'分块执行预计需约{gpu_estimate/2**30:.1f} GiB可用显存，当前{free/2**30:.1f}。')
    require(psutil.virtual_memory().available > host_estimate,
            f'CPU权重和缓冲区预计需约{host_estimate/2**30:.1f} GiB可用主内存，请先关闭占内存程序。')
    if training:
        # 工作梯度+m/v，以及首次完整断点及其替换临时副本。最终已完成权重由外层另存。
        needed = model_parameter_count*4*9 + 2**30
        require(shutil.disk_usage('step8').free > needed, '空间不足以保存分块工作状态和原子断点。')
    return dict(backend=BACKEND,chunk_bytes=CHUNK_BYTES,gpu_estimate_gib=gpu_estimate/2**30,
                host_estimate_gib=host_estimate/2**30,optimizer='PyTorch float32 Adam, foreach=False',
                batch_partitioned=False,linear_partition='none: complete GEMM per layer',TF32=False,AMP=False)


def small_test():
    """小网络比较常规CUDA与分层CUDA的梯度、Adam和多步结果；另测磁盘断点。

    仅质量检查，生成文件放在quality_control，不能作为正式模型结果使用。
    """
    import step8_experiments as core
    device=core.runtime('cuda')
    folder=ensure_folder('step8/quality_control/streaming_small_test')
    config=copy.deepcopy(core.condition_catalog()[0]['deep'])
    config['hidden_dims']=[23,19,17,13]
    config['latent_dim']=7
    core.seed_all(629108)
    base=core.Autoencoder(config,37)
    reference=copy.deepcopy(base).to(device)
    disk=DiskState(folder/'work',base.named_parameters())
    model=install_streaming(base,device,disk,chunk_elements=71)
    optimizer=StreamedAdam(model,disk,device,chunk_elements=71)
    ordinary=torch.optim.Adam(reference.parameters(),lr=.0002,foreach=False)
    matrix=np.random.default_rng(629108).normal(size=(128,37)).astype(np.float16)
    measurements=[]
    # 同一输入、初始参数与Dropout RNG；元素chunk=71强制所有主要权重分块。
    for step in range(3):
        x=torch.from_numpy(matrix[step%2*64:(step%2+1)*64].astype(np.float32)).to(device)
        start_rng=core.rng_state()
        ordinary.zero_grad(set_to_none=True)
        reference.train();_,out_reference=reference(x)
        loss_reference=nn.functional.mse_loss(out_reference,x);loss_reference.backward()
        reference_grad={n:p.grad.detach().clone() for n,p in reference.named_parameters()}
        after_reference_rng=core.rng_state()
        core.restore_rng(start_rng)
        optimizer.zero_grad();model.train();_,out_model=model(x)
        loss_model=nn.functional.mse_loss(out_model,x);loss_model.backward()
        max_gradient=0.
        for name,p in model.named_parameters():
            if name in disk.ready:
                gradient=disk.read_gpu(name,'gradient',0,p.numel(),device).reshape(p.shape)
            else:
                gradient=p.grad
            torch.testing.assert_close(gradient,reference_grad[name],rtol=3e-4,atol=3e-5)
            max_gradient=max(max_gradient,float((gradient-reference_grad[name]).abs().max()))
        torch.testing.assert_close(out_model,out_reference,rtol=3e-5,atol=3e-5)
        require(torch.equal(torch.cuda.get_rng_state(),after_reference_rng['cuda'][device.index]),
                'Dropout RNG消耗与常规执行不同。')
        ordinary.step();optimizer.step()
        max_parameter=0.
        for name,p in model.named_parameters():
            expected=dict(reference.named_parameters())[name]
            error=float((p.detach().to(device)-expected.detach()).abs().max())
            max_parameter=max(max_parameter,error)
            # 浮点零附近的BN前bias可能由Adam放大舍入，按非退化输出/梯度另核验。
            torch.testing.assert_close(p.detach().to(device),expected.detach(),rtol=3e-4,atol=3e-4)
            for kind,ordinary_key in (('mean','exp_avg'),('variance','exp_avg_sq')):
                moment=disk.read_gpu(name,kind,0,p.numel(),device).reshape(p.shape)
                torch.testing.assert_close(moment,ordinary.state[expected][ordinary_key],rtol=5e-4,atol=2e-5)
        measurements.append(dict(step=step+1,loss_reference=float(loss_reference.detach()),loss_streamed=float(loss_model.detach()),
            max_abs_gradient_difference=max_gradient,max_abs_parameter_difference=max_parameter,
            max_abs_output_difference=float((out_model-out_reference).detach().abs().max())))
        del reference_grad
    # 保存真实磁盘moment，随后用有界窗口恢复到另一份工作缓存。
    checkpoint=folder/'resume_test.pt'
    with optimizer.checkpoint_state() as state:
        core.save_torch(dict(model_state_dict=model.state_dict(),optimizer_state_dict=state,rng=core.rng_state()),checkpoint)
    restored=checkpoint_metadata(checkpoint)
    resumed=core.Autoencoder(config,37)
    load_model_checkpoint(resumed,checkpoint,restored['model_state_dict'])
    resumed_disk=DiskState(folder/'restored_work',resumed.named_parameters())
    install_streaming(resumed,device,resumed_disk,chunk_elements=71)
    resumed_optimizer=StreamedAdam(resumed,resumed_disk,device,chunk_elements=71)
    resumed_optimizer.load_state_dict(restored['optimizer_state_dict'],checkpoint)
    restart_rng=materialize_metadata(checkpoint,restored['rng']);del restored;gc.collect()
    # 用同一输入、同一RNG各执行下一次更新，要求恢复轨迹逐位一致。
    next_x=torch.from_numpy(matrix[:64].astype(np.float32)).to(device)
    for net,opt in ((model,optimizer),(resumed,resumed_optimizer)):
        core.restore_rng(restart_rng);net.train();opt.zero_grad()
        _,output=net(next_x);loss=nn.functional.mse_loss(output,next_x);loss.backward();opt.step()
    for name,value in model.state_dict().items():
        require(torch.equal(value.cpu(),resumed.state_dict()[name].cpu()),f'恢复参数/BN不完全一致：{name}')
    require(optimizer.steps==resumed_optimizer.steps,'恢复Adam step不一致。')
    for name,p in model.named_parameters():
        for kind in ('mean','variance'):
            require(core.sha(disk.path(name,kind))==core.sha(resumed_disk.path(name,kind)),
                    f'恢复Adam moment不一致：{name}/{kind}')
    # 确认所有读取窗口已关闭，Windows下可原子替换同一个checkpoint。
    with resumed_optimizer.checkpoint_state() as state:
        core.save_torch(dict(model_state_dict=resumed.state_dict(),optimizer_state_dict=state),checkpoint)
    result=dict(status='passed',backend=BACKEND,device=str(device),batch_size=64,
        comparisons=measurements,exact_streamed_resume=True,atomic_checkpoint_replacement=True,
        full_batch_bn=True,dropout_rng_preserved=True,versions=core.versions(),
        backend_sha256=core.sha(Path('step8/code/streamed_training.py')))
    core.write_json(folder/'report.json',result)
    disk.remove_work_files();resumed_disk.remove_work_files();checkpoint.unlink()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def full_smoke():
    """真实深层维度与batch64的完整更新、eval、保存/恢复和权重传播检查。

    只用训练划分中的前64个真实样本执行两次诊断更新，不运行正式epoch。
    所有巨型临时权重/Adam文件在验证成功后删除，正式模型目录不写入。
    """
    import step8_experiments as core
    import psutil
    folder=ensure_folder('step8/quality_control/streaming_full_smoke')
    device=core.runtime('cuda')
    core.verify_setup()
    config=copy.deepcopy(core.condition_catalog()[0]['deep'])
    core.memory_preflight(config,60498,device)
    timings={};started=time.monotonic();phase_start=started
    def mark(phase):
        nonlocal phase_start
        torch.cuda.synchronize(device)
        timings[phase]=time.monotonic()-phase_start
        core.write_json(folder/'progress.json',dict(status='running',last_completed=phase,
            elapsed_seconds=time.monotonic()-started,phases=timings,
            gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
            host_process_rss_gib=psutil.Process().memory_info().rss/2**30))
        print(f'full-smoke {phase}: {timings[phase]:.1f}s; peak GPU {torch.cuda.max_memory_allocated(device)/2**30:.3f} GiB',flush=True)
        phase_start=time.monotonic()
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    core.seed_all(6298108)
    matrix=np.load(core.MATRIX,mmap_mode='r',allow_pickle=False)
    _,indices,_=core.sample_inputs()
    values=np.asarray(matrix[indices[:64]],dtype=np.float32).copy();del matrix
    x=torch.from_numpy(values).to(device)
    model=core.Autoencoder(config,60498)
    count=sum(p.numel() for p in model.parameters())
    disk=DiskState(folder/'work',model.named_parameters())
    install_streaming(model,device,disk)
    optimizer=StreamedAdam(model,disk,device)
    mark('initialization')
    optimizer.zero_grad();model.train()
    training_latent,reconstruction=model(x)
    loss=nn.functional.mse_loss(reconstruction,x)
    require(bool(torch.isfinite(loss)),'完整模型的诊断损失不是有限值。')
    loss_value=float(loss.detach());loss.backward()
    del training_latent,reconstruction,loss
    mark('forward_backward')
    optimizer.step()
    mark('adam_update')
    model.eval()
    with torch.no_grad():
        latent,prediction=model(x)
    require(prediction.shape==(64,60498) and latent.shape==(64,64),'深层维度错误。')
    require(bool(torch.isfinite(prediction).all()),'更新后输出含非有限值。')
    predicted=prediction.cpu().numpy().copy();encoded=latent.cpu().numpy().copy()
    del latent,prediction
    mark('evaluation')
    checkpoint=folder/'resume_smoke.pt'
    with optimizer.checkpoint_state() as state:
        core.save_torch(dict(model_state_dict=model.state_dict(),optimizer_state_dict=state,
            progress=dict(epoch=1,phase='train',cursor=64,order=torch.arange(6289)),
            rng=core.rng_state()),checkpoint)
    mark('atomic_save')
    # 在释放模型前记录每个moment文件的指纹，恢复后逐文件核对。
    moment_hashes={(name,kind):core.sha(disk.path(name,kind)) for name in disk.names for kind in ('mean','variance')}
    del optimizer,model,disk;gc.collect();torch.cuda.empty_cache()
    saved=checkpoint_metadata(checkpoint)
    with torch.device('meta'):
        model=core.Autoencoder(config,60498)
    model.to_empty(device='cpu')
    load_model_checkpoint(model,checkpoint,saved['model_state_dict'])
    disk=DiskState(folder/'work',model.named_parameters())
    install_streaming(model,device,disk)
    optimizer=StreamedAdam(model,disk,device)
    optimizer.load_state_dict(saved['optimizer_state_dict'],checkpoint)
    progress=materialize_metadata(checkpoint,saved['progress'])
    core.restore_rng(materialize_metadata(checkpoint,saved['rng']))
    del saved;gc.collect()
    require(optimizer.steps==1 and progress['cursor']==64,'恢复进度/Adam step错误。')
    for (name,kind),expected in moment_hashes.items():
        require(core.sha(disk.path(name,kind))==expected,f'完整moment恢复指纹不同：{name}/{kind}')
    mark('checkpoint_restore_and_verify')
    model.eval()
    with torch.no_grad():
        latent,prediction=model(x)
    require(np.array_equal(prediction.cpu().numpy(),predicted) and np.array_equal(latent.cpu().numpy(),encoded),
            '恢复后的完整模型输出不完全相同。')
    del latent,prediction,predicted,encoded
    with optimizer.checkpoint_state() as state:
        core.save_torch(dict(model_state_dict=model.state_dict(),optimizer_state_dict=state,progress=progress),checkpoint)
    mark('restored_evaluation_and_atomic_replace')
    # 第二次更新实际读取已经恢复的磁盘moment，覆盖非零Adam状态的显存路径。
    optimizer.zero_grad();model.train()
    latent,reconstruction=model(x)
    resumed_loss=nn.functional.mse_loss(reconstruction,x)
    require(bool(torch.isfinite(resumed_loss)), '接续批次的损失不是有限值。')
    resumed_loss_value=float(resumed_loss.detach())
    resumed_loss.backward();optimizer.step()
    del latent,reconstruction,resumed_loss
    require(optimizer.steps==2, '接续后Adam step未正确递增。')
    mark('resumed_forward_backward_adam')
    model.eval()
    model.requires_grad_(False)
    entry,scores,audit=core.propagate_weight(model.decoder)
    require(entry.shape==(64,256) and scores.shape==(64,60498),'深层权重传播形状错误。')
    mark('weight_propagation')
    result=dict(status='passed',backend=BACKEND,batch_size=64,n_parameters=count,input_dim=60498,
        latent_dim=64,n_diagnostic_optimizer_steps=2,seed=6298108,
        scientific_output=False,loss_before_update=loss_value,resumed_batch_loss=resumed_loss_value,phases_seconds=timings,
        gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        gpu_name=torch.cuda.get_device_name(device),host_process_peak_working_set_gib=
            getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss)/2**30,
        exact_restored_predictions=True,exact_restored_moment_files=True,atomic_replacement=True,
        propagation_audit=audit,versions=core.versions(),
        backend_sha256=core.sha(Path('step8/code/streamed_training.py')),
        core_sha256=core.sha(core.CODE))
    checkpoint.unlink();disk.remove_work_files()
    result['large_temporary_files_removed']=True
    core.write_json(folder/'report.json',result)
    core.write_json(folder/'progress.json',dict(status='complete',phases=timings))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    import step8_experiments as core
    core.enter_trial()
    parser=argparse.ArgumentParser(description='Validate the Step8 full-precision CUDA streaming backend')
    parser.add_argument('stage',choices=['small-test','full-smoke'])
    args=parser.parse_args()
    # 与正式训练使用同一排他锁，避免验证和训练同时争用GPU/写入资源。
    with core.training_lock():
        if args.stage=='small-test':small_test()
        elif args.stage=='full-smoke':full_smoke()


if __name__=='__main__':
    main()
