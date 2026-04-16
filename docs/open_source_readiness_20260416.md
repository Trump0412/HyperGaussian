# Open-Source Readiness Audit (2026-04-16)

## 审计范围

- 仓库：`/root/autodl-tmp/HyperGaussian`
- 目标：判断是否可直接上传 GitHub 开源，并给出整改优先级。

## 结论

当前状态是 **“接近可开源，但不建议直接一键公开”**。

核心原因不是算法不可复现，而是发布层面的两个关键项尚未最终落地：

1. 许可证文件（`LICENSE`）尚未确定。
2. 仓库中仍有较多历史实验脚本与未跟踪内容，缺少一次“对外发布裁剪”。

## 已完成项

- README 已重写为可执行流程版本（环境、数据、模型、复现、benchmark）。
- keyboard 固定配置与目标指标（PSNR 28.4051）已在文档中锁定。
- OursBench 与 4DLangSplat/Americano 的复现流程已落文档，并有本地复现记录。
- GitHub Pages 首页模板已补齐（`docs/index.html` + `docs/assets/githubio.css`）。

## 仍需处理（发布前）

### P0（必须）

1. 许可证确认并落库
- 建议在发布前由负责人确定 `MIT` / `Apache-2.0` / 其他许可证，并写入仓库根目录 `LICENSE`。

2. 对外发布文件集冻结
- 清理一次性实验脚本（保留“支持脚本白名单”）。
- 确认 `external/` 策略：
  - 方案 A：作为 submodule；
  - 方案 B：保留 vendor 代码并明确版本来源；
  - 方案 C：通过 bootstrap 脚本自动拉取。

### P1（建议）

1. 统一 benchmark 评测口径
- 当前历史结果与新评测脚本口径存在差异，建议在 release note 中固定“官方口径版本”。

2. 添加 `RELEASE_NOTES.md`
- 明确版本号、评测口径、已知限制、数据下载链接。

## 建议的发布前 checklist

- [ ] `LICENSE` 已确认。
- [ ] `README` 里的所有命令在干净环境可跑通。
- [ ] benchmark JSON / query 映射文件均有公开下载地址。
- [ ] `external/` 依赖获取方式在 README 里唯一且清晰。
- [ ] 首页链接（GitHub/HG/arXiv）替换为正式地址。

