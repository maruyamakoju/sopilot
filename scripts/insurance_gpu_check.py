#!/usr/bin/env python3
"""Insurance MVP GPU Readiness Check.

Checks whether the system is ready to run real Video-LLM inference:
1. GPU detection (CUDA availability)
2. VRAM check (14GB minimum for Qwen2.5-VL-7B)
3. Dependency packages (transformers, qwen-vl-utils)
4. Model download status
5. Mock vs Real backend status

Usage:
    python scripts/insurance_gpu_check.py
    python scripts/insurance_gpu_check.py --json
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_gpu() -> dict:
    """Check GPU availability and VRAM."""
    result = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "sufficient_vram": False,
        "min_vram_gb": 14.0,
    }

    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_version"] = torch.__version__

        if result["cuda_available"]:
            result["gpu_count"] = torch.cuda.device_count()
            for i in range(result["gpu_count"]):
                props = torch.cuda.get_device_properties(i)
                total_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)
                result["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(total_gb, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
                if total_gb >= result["min_vram_gb"]:
                    result["sufficient_vram"] = True

            result["cuda_version"] = torch.version.cuda
        else:
            result["torch_version"] = torch.__version__
    except ImportError:
        result["torch_installed"] = False
        result["error"] = "PyTorch not installed"

    return result


def check_dependencies() -> dict:
    """Check required Python packages."""
    deps = {
        "torch": {"installed": False, "version": None, "required": True},
        "transformers": {"installed": False, "version": None, "required": True, "min_version": "4.45.0"},
        "qwen_vl_utils": {"installed": False, "version": None, "required": True},
        "cv2": {"installed": False, "version": None, "required": True},
        "PIL": {"installed": False, "version": None, "required": True},
        "pydantic": {"installed": False, "version": None, "required": True},
    }

    try:
        import torch
        deps["torch"]["installed"] = True
        deps["torch"]["version"] = torch.__version__
    except ImportError:
        pass

    try:
        import transformers
        deps["transformers"]["installed"] = True
        deps["transformers"]["version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import qwen_vl_utils
        deps["qwen_vl_utils"]["installed"] = True
        deps["qwen_vl_utils"]["version"] = getattr(qwen_vl_utils, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import cv2
        deps["cv2"]["installed"] = True
        deps["cv2"]["version"] = cv2.__version__
    except ImportError:
        pass

    try:
        import PIL
        from PIL import Image  # noqa: F401
        deps["PIL"]["installed"] = True
        deps["PIL"]["version"] = PIL.__version__
    except ImportError:
        pass

    try:
        import pydantic
        deps["pydantic"]["installed"] = True
        deps["pydantic"]["version"] = pydantic.__version__
    except ImportError:
        pass

    deps["all_required_installed"] = all(
        d["installed"] for d in deps.values() if d["required"]
    )

    return deps


def check_model_download() -> dict:
    """Check if Qwen2.5-VL-7B model is downloaded."""
    result = {
        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
        "downloaded": False,
        "cache_path": None,
    }

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if "Qwen2.5-VL-7B" in repo.repo_id:
                result["downloaded"] = True
                result["cache_path"] = str(repo.repo_path)
                result["size_gb"] = round(repo.size_on_disk / (1024 ** 3), 2)
                break
    except ImportError:
        result["huggingface_hub_installed"] = False
    except Exception as e:
        result["error"] = str(e)

    return result


def check_backend_status() -> dict:
    """Check current VLM backend configuration."""
    result = {
        "current_backend": "mock",
        "available_backends": ["mock"],
        "recommendation": "",
    }

    import os
    backend_env = os.getenv("INSURANCE_VLM_BACKEND", os.getenv("INSURANCE_COSMOS_BACKEND", "mock"))
    result["current_backend"] = backend_env

    # Check what backends are available
    try:
        from insurance_mvp.cosmos.client import VideoLLMClient, VLMConfig
        result["available_backends"].append("qwen2.5-vl-7b")

        # Try health check if available
        try:
            config = VLMConfig(model_name="mock")
            client = VideoLLMClient(config)
            health = client.health_check() if hasattr(client, "health_check") else {"status": "unknown"}
            result["mock_health"] = health
        except Exception:
            pass
    except ImportError:
        pass

    # Recommendation
    gpu = check_gpu()
    deps = check_dependencies()
    if gpu["sufficient_vram"] and deps["all_required_installed"]:
        result["recommendation"] = "Ready for real inference. Set INSURANCE_VLM_BACKEND=qwen2.5-vl-7b"
    elif gpu["cuda_available"] and not gpu["sufficient_vram"]:
        result["recommendation"] = "GPU detected but insufficient VRAM (need 14GB+). Use mock or smaller model."
    elif not gpu["cuda_available"]:
        result["recommendation"] = "No GPU detected. Use mock backend or install CUDA."
    else:
        result["recommendation"] = "Install missing dependencies first."

    return result


def run_check(as_json: bool = False) -> dict:
    """Run all checks and display results."""
    results = {
        "gpu": check_gpu(),
        "dependencies": check_dependencies(),
        "model": check_model_download(),
        "backend": check_backend_status(),
    }

    # Overall readiness
    results["ready_for_real_inference"] = (
        results["gpu"]["sufficient_vram"]
        and results["dependencies"]["all_required_installed"]
        and results["model"]["downloaded"]
    )

    if as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("=" * 60)
        print("Insurance MVP GPU Readiness Check")
        print("=" * 60)

        # GPU
        print("\n--- GPU ---")
        gpu = results["gpu"]
        if gpu["cuda_available"]:
            print(f"  CUDA: Available (version {gpu.get('cuda_version', 'unknown')})")
            print(f"  PyTorch: {gpu.get('torch_version', 'unknown')}")
            print(f"  GPUs found: {gpu['gpu_count']}")
            for g in gpu["gpus"]:
                vram_ok = "OK" if g["total_memory_gb"] >= 14.0 else "INSUFFICIENT"
                print(f"    [{g['index']}] {g['name']}: {g['total_memory_gb']} GB ({vram_ok})")
            print(f"  Sufficient VRAM (>=14GB): {'YES' if gpu['sufficient_vram'] else 'NO'}")
        else:
            print("  CUDA: Not available")
            if "error" in gpu:
                print(f"  Error: {gpu['error']}")

        # Dependencies
        print("\n--- Dependencies ---")
        deps = results["dependencies"]
        for name, info in deps.items():
            if name == "all_required_installed":
                continue
            status = "OK" if info["installed"] else "MISSING"
            ver = f" ({info['version']})" if info["version"] else ""
            req = " [REQUIRED]" if info.get("required") else ""
            print(f"  {name}: {status}{ver}{req}")

        # Model
        print("\n--- Model Download ---")
        model = results["model"]
        if model["downloaded"]:
            print(f"  {model['model_name']}: Downloaded ({model.get('size_gb', '?')} GB)")
        else:
            print(f"  {model['model_name']}: NOT DOWNLOADED")
            print(f"  Download: python -c \"from transformers import AutoProcessor; AutoProcessor.from_pretrained('{model['model_name']}')\"")

        # Backend
        print("\n--- Backend Status ---")
        backend = results["backend"]
        print(f"  Current: {backend['current_backend']}")
        print(f"  Available: {', '.join(backend['available_backends'])}")

        # Overall
        print("\n--- Overall ---")
        if results["ready_for_real_inference"]:
            print("  READY for real Video-LLM inference")
        else:
            print("  NOT READY for real inference")
        print(f"  Recommendation: {backend['recommendation']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Insurance MVP GPU Readiness Check")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = run_check(as_json=args.json)

    if not results["ready_for_real_inference"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
