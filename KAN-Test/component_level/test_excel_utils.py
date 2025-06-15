import pytest
import io
from fastapi import UploadFile, HTTPException
from common_utils import file
import pandas as pd

class DummyUploadFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.file = io.BytesIO(content)
    async def read(self):
        self.file.seek(0)
        return self.file.read()
    async def close(self):
        self.file.close()

@pytest.mark.asyncio
async def test_read_excel_file_valid():
    # 创建一个简单的Excel文件
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    upload = DummyUploadFile('test.xlsx', buf.read())
    buf.seek(0)
    upload.file = io.BytesIO(buf.read())
    result = await file.read_excel_file(upload)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

@pytest.mark.asyncio
async def test_read_excel_file_empty():
    upload = DummyUploadFile('empty.xlsx', b'')
    with pytest.raises(HTTPException) as exc:
        await file.read_excel_file(upload)
    assert (
        '未提供文件' in str(exc.value.detail)
        or 'Excel文件为空' in str(exc.value.detail)
        or 'Excel file format cannot be determined' in str(exc.value.detail)
    )

@pytest.mark.asyncio
async def test_read_excel_file_invalid_ext():
    upload = DummyUploadFile('test.txt', b'123')
    with pytest.raises(HTTPException) as exc:
        await file.read_excel_file(upload)
    assert '只支持Excel文件' in str(exc.value.detail) 