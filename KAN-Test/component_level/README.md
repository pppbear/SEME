# 组件级（Component Level）单元测试说明

本目录包含对项目各服务（predict_service、data_service、compare_service、analyze_service、auth_service、api_gateway 及通用模块）的组件级（单元）测试，使用 [pytest](https://docs.pytest.org/) 进行自动化测试。

---

## 一、如何运行所有测试

1. **安装依赖**（建议在虚拟环境中）：
   ```bash
   pip install -r ../../environment.yml
   # 或 conda env create -f ../../environment.yml
   ```

2. **运行所有测试**：
   ```bash
   pytest .
   ```

3. **生成覆盖率报告**（可选）：
   ```bash
   pytest --cov=../../common_utils --cov=../../common_db --cov=../../predict_service --cov=../../data_service --cov=../../compare_service --cov=../../analyze_service --cov=../../auth_service --cov=../../api_gateway .
   ```

---

## 二、测试内容覆盖范围

- **通用工具/数据库模块**：如 `common_utils/preprocess.py`、`common_db/utils/db_utils.py`、`common_db/utils/auth_utils.py`、`common_utils/kan/utils.py` 等。
- **predict_service**：模型预测、特征处理、模型缓存等核心函数。
- **data_service**：数据读取、数据校验、数据转换等函数。
- **compare_service**：结果比对、统计分析等核心算法。
- **analyze_service**：数据分析、特征提取、统计计算等函数。
- **auth_service**：用户认证、权限校验、token处理、用户CRUD等。
- **api_gateway**：路由分发、参数校验、聚合逻辑等。

每个服务的主要业务逻辑、工具函数、数据结构等均应有对应的单元测试。

---

## 三、如何扩展/补充测试

1. **新建测试文件**：以 `test_xxx.py` 命名，放在本目录下。
2. **编写测试用例**：每个函数/类/模块建议有独立的测试函数，使用 `pytest` 风格（以 `test_` 开头）。
3. **Mock 外部依赖**：如需隔离数据库、API等外部依赖，可用 `pytest` 的 fixture 或 `monkeypatch`。
4. **运行并检查**：确保所有测试通过，无异常。

---

## 四、注意事项

- 测试应覆盖正常、异常、边界等多种情况。
- 避免直接操作生产数据库，可用 sqlite 或 mock 数据。
- 如需测试API路由，建议用 FastAPI 的 TestClient。

---

如需补充具体模块/函数的测试用例模板，请联系开发负责人或参考已有测试文件。 