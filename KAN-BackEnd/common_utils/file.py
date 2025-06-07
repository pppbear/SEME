import pandas as pd
from fastapi import UploadFile, HTTPException, status
from typing import List
import os
import tempfile
import uuid
from datetime import datetime

async def read_excel_file(file: UploadFile) -> pd.DataFrame:
    """
    读取上传的Excel文件到pandas DataFrame
    
    Args:
        file (UploadFile): 上传的Excel文件
        
    Returns:
        pd.DataFrame: 读取的数据
        
    Raises:
        HTTPException: 当文件读取失败时抛出异常
    """
    if not file.file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="未提供文件"
        )
        
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="文件名不能为空"
        )
        
    # 检查文件扩展名
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.xlsx', '.xls']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="只支持Excel文件(.xlsx, .xls)"
        )
    
    # 生成唯一的临时文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    temp_filename = f"compare_service_{timestamp}_{unique_id}{file_ext}"
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, prefix=f"compare_service_{timestamp}_{unique_id}_") as temp_file:
            print(f"创建临时文件: {temp_file.name}")
            
            # 将上传的文件内容写入临时文件
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # 使用pandas读取临时文件
            df = pd.read_excel(temp_file.name)
            
            if df.empty:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, 
                    detail="Excel文件为空"
                )
                
            return df
            
    except pd.errors.EmptyDataError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Excel文件为空"
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Excel文件格式错误"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"读取Excel文件失败: {str(e)}"
        )
    finally:
        # 清理临时文件
        try:
            if 'temp_file' in locals():
                print(f"删除临时文件: {temp_file.name}")
                os.unlink(temp_file.name)
        except Exception as e:
            print(f"删除临时文件失败: {str(e)}")
        # 确保文件被关闭
        await file.close() 