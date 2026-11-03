import torch
from typing import Dict, Any


def unwrap_model(model):
    """Unwrap model from DistributedDataParallel if needed."""
    if hasattr(model, 'module'):
        return model.module
    return model

def get_attr_by_path(obj, path):
    for attr in path.split("."):
        if not hasattr(obj, attr):
            return None
        obj = getattr(obj, attr)
    return obj

def calculate_module_grad_norm(module: torch.nn.Module) -> Dict[str, float]:
    """Calculate gradient norm and max gradient for a module."""
    total_norm_sq = 0.0
    max_grad = 0.0
    param_count = 0

    for param in module.parameters():
        if param.grad is not None:
            grad = param.grad.detach().float()
            total_norm_sq += grad.pow(2).sum().item()
            max_grad = max(max_grad, grad.abs().max().item())
            param_count += 1

    if param_count == 0:
        return {'grad_norm': 0.0, 'max_grad': 0.0, 'param_count': 0}

    grad_norm = total_norm_sq ** 0.5

    return {
        'grad_norm': float(grad_norm),
        'max_grad': float(max_grad),
        'param_count': param_count
    }

def calculate_parameter_grad_norm(param: Any) -> Dict[str, float]:
    """Calculate gradient stats for a single parameter/tensor target."""
    if param is None or not torch.is_tensor(param):
        return {'grad_norm': 0.0, 'max_grad': 0.0, 'param_count': 0}
    if param.grad is None:
        return {'grad_norm': 0.0, 'max_grad': 0.0, 'param_count': 1}

    grad = param.grad.detach().float()
    return {
        'grad_norm': float(grad.norm().item()),
        'max_grad': float(grad.abs().max().item()),
        'param_count': 1
    }

def monitor_all_module_gradients(model, modules_to_monitor) -> Dict[str, Dict[str, float]]:
    """Monitor gradients for all major model components."""
    unwrapped_model = unwrap_model(model)
    results = {}

    # Monitor modules (skip paths that don't exist on the current architecture)
    for module_name in modules_to_monitor:
        target = get_attr_by_path(unwrapped_model, module_name)
        if target is None:
            results[module_name] = {'grad_norm': 0.0, 'max_grad': 0.0, 'param_count': 0}
            continue

        if isinstance(target, torch.nn.Module):
            results[module_name] = calculate_module_grad_norm(target)
        elif isinstance(target, torch.nn.Parameter) or torch.is_tensor(target):
            results[module_name] = calculate_parameter_grad_norm(target)
        else:
            results[module_name] = {'grad_norm': 0.0, 'max_grad': 0.0, 'param_count': 0}

    return results


def monitor_all_gradients(model, modules_to_monitor) -> Dict[str, Any]:
    """Monitor gradients for all model components"""
    module_results = monitor_all_module_gradients(model, modules_to_monitor)

    return {
        'modules': module_results
    }


def get_gradient_summary(model, modules_to_monitor) -> Dict[str, float]:
    """Get a flat summary of gradient norms for logging."""
    all_grad_info = monitor_all_gradients(model, modules_to_monitor)
    grad_info = all_grad_info['modules']
    summary = {}

    # Flatten encoder gradients
    for key, stats in grad_info.items():
        summary[f'{key}_grad_norm'] = stats['grad_norm']
        summary[f'{key}_max_grad'] = stats['max_grad']

    return summary


def log_module_gradients(model, step: int, to_console: bool = False, modules_to_monitor: list = None):
    """Log gradient information for all model modules."""
    if modules_to_monitor is None:
        modules_to_monitor = []

    summary = get_gradient_summary(model, modules_to_monitor)

    if to_console:
        print(f"\n=== Module Gradients - Step {step} ===")

        # Print encoder gradients
        for module in modules_to_monitor:
            grad_norm_key = f'{module}_grad_norm'
            max_grad_key = f'{module}_max_grad'
            print(
                f"{module:30s}: "
                f"norm={summary[grad_norm_key]:.6f}, "
                f"max={summary[max_grad_key]:.6f}"
            )

        print("=" * 50)

    return summary
