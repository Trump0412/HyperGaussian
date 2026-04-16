# GitHub Publish Steps

## 0) 先决条件

- 在 GitHub 网页端创建一个**空仓库**（不要勾选 Initialize with README）。
- 记录仓库地址，例如：
  - HTTPS: `https://github.com/<USER>/<REPO>.git`
  - SSH: `git@github.com:<USER>/<REPO>.git`

## 1) 进入项目

```bash
cd /root/autodl-tmp/HyperGaussian
```

## 2) 仅添加本次开源整理文件（避免误提交大目录）

```bash
git add \
  README.md \
  docs/index.html \
  docs/assets/githubio.css \
  docs/open_source_readiness_20260416.md \
  docs/reproducibility_20260416.md \
  docs/github_publish_steps.md \
  scripts/common.sh \
  scripts/setup_baseline_env.sh \
  scripts/setup_grounded_sam2.sh \
  scripts/setup_gsam2_env.sh \
  scripts/run_public_query_protocol.sh \
  scripts/list_public_protocol_queries.py \
  scripts/export_entitybank.sh \
  scripts/render_stellar_tube.sh \
  scripts/run_query_guided_full.sh
```

## 3) 提交

```bash
git commit -m "release: prepare open-source docs, reproducibility, and publish-ready scripts"
```

## 4) 绑定远端并推送

```bash
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

如果你已经有 `origin`：

```bash
git remote set-url origin <YOUR_REPO_URL>
git push -u origin main
```

## 5) 开启 GitHub Pages

- 仓库 `Settings -> Pages`
- `Source` 选择 `Deploy from a branch`
- Branch 选择 `main`，目录选择 `/docs`

随后主页会发布在：

`https://<USER>.github.io/<REPO>/`

## 6) 发布后需要替换的占位符

- `docs/index.html`:
  - `https://github.com/<ORG>/<REPO>`
  - `https://<HG-LINK>`
  - `https://arxiv.org/abs/<ARXIV-ID>`
- `README.md`:
  - `https://huggingface.co/datasets/<ORG>/GaussianStellar-OursBench`

