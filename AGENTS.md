# 濠梁项目说明

濠梁是本地读者模型，当前核心链路是术语账本、读者切片、草稿注解和改稿。先看 `README.md`、`schema/term.schema.json` 与 `config/reader.json`。

- 每张术语卡对应一个概念，掌握度可追溯到证据。
- 层级、权重和衰减从配置读取。领域是检索排序先验。
- 归档、合并和写入须维护账本一致性。模型生成的证据通过候选卡记录。
- 示例和测试使用合成数据。真实账本、个人资料与凭据留在本地。
- 提交前执行 `python -m pytest -q` 与 `python -m ruff check src tests eval`。
