# 基线问卷

[Java 后端示例问卷](baseline-java-2026-11-22.md)覆盖 8 个领域、82 个术语。每项填写 0 到 3，分别对应「不知道」「听过」「会用」「能讲」。按领域分批填写时，空白行会被跳过。

从项目根目录导入：

```bash
PYTHONPATH=src python eval/import_baseline.py \
  docs/onboarding/baseline-java-2026-11-22.md \
  --store var/reader
```

导入器从相邻 JSON 文件读取领域映射和自评层级。重新导入会替换旧的自评证据，已含其他证据的卡片会保留。问卷模板可以复制到本地再填写。
