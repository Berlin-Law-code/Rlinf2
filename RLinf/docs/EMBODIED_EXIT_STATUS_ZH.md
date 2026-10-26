# Embodied 训练入口的错误退出状态

`examples/embodiment/train_embodied_agent.py` 的主进程发生异常时，启动脚本必须获得非零退出码。即使采样、优化器更新、保存模型或末尾评测已经完成，之后的日志汇总、计时或清理异常仍表示本次运行未正常完成。

Cluster 目前在正常解释器退出时使用 `os._exit(0)` 快速结束，避免已知的退出清理挂起。它的 `sys.excepthook` 会标记未处理异常，但 Python 的 `SystemExit` 不经过这个 hook。Hydra 可能先捕获任务异常，输出错误，再抛出 `SystemExit(1)`；如果入口没有额外处理，Cluster 会误走成功路径，把失败进程的退出码改成 0。

入口现在在 Hydra 包装后的 `main()` 外捕获异常：

- 非正常 `SystemExit` 先设置 `Cluster._run_failed=True`，随后原样重新抛出，保留原来的非零退出码和错误输出。
- 其他 `BaseException` 同样标记失败并重新抛出，包括普通未处理异常和中断。
- 正常返回、`SystemExit(None)` 和整数 `SystemExit(0)` 保持成功退出。非整数退出对象遵循 Python 的错误退出语义，例如 `SystemExit("错误")` 和 `SystemExit(0.0)` 均非成功。

这项修改只覆盖该 embodied 入口，保留 Cluster 正常退出的快速路径和已有失败清理路径；没有全局删除 `os._exit(0)`，也没有吞掉异常。其他入口若存在相同的包装异常路径，需要单独审查。

验证见 `tests/unit_tests/test_embodied_entrypoint_exit_status.py`：真实 CPU 子进程执行入口的生产异常边界及 Cluster 的生产 hook/atexit 方法。测试通过 AST 排除模型与调度器初始化，不导入 Torch、Ray 或 TensorFlow。Hydra 测试使用真正的 Hydra 包装器，验证任务异常被转换为非零 `SystemExit` 后，仍返回非零状态；另有旧边界对照，复现失败被改写为 0 的问题。

运行命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/unit_tests/test_embodied_entrypoint_exit_status.py
```

控制脚本应保存并传播入口进程的退出码。退出码 0 仍只是正常结束的必要检查之一；完整验证还应确认预期训练轮次、优化器更新数、评测记录及错误日志，不能仅凭退出码宣称训练效果改善。
