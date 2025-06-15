import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime, timedelta

from auth_service.app.core.config import settings
from auth_service.app.utils.redis_client import redis_client

# 存储验证码的字典，格式: {email: {"code": "123456", "type": "register", "expiry": datetime}}
verification_codes: Dict[str, Dict[str, any]] = {}

def generate_verification_code(length: int = 6) -> str:
    """生成指定长度的随机数字验证码"""
    return ''.join(random.choices(string.digits, k=length))

def save_verification_code(email: str, code: str, code_type: str) -> None:
    """保存验证码到Redis，并设置过期时间"""
    key = f"verify:{code_type}:{email}"
    redis_client.setex(key, 600, code)

def verify_code(email: str, provided_code: str, code_type: str) -> bool:
    """验证用户提供的验证码是否正确"""
    key = f"verify:{code_type}:{email}"
    stored_code = redis_client.get(key)
    if stored_code is None:
        return False
    if stored_code != provided_code:
        return False
    # 校验成功后删除验证码，防止复用
    redis_client.delete(key)
    return True

def send_email(recipient: str, subject: str, body: str) -> bool:
    """发送邮件的通用函数"""
    try:
        # 创建邮件对象
        msg = MIMEMultipart()
        msg['From'] = settings.EMAIL_SENDER
        msg['To'] = recipient
        msg['Subject'] = subject
        
        # 添加邮件正文
        msg.attach(MIMEText(body, 'html'))
        
        # 使用SSL连接
        server = smtplib.SMTP_SSL(settings.SMTP_SERVER, settings.SMTP_PORT)
        
        # 登录
        if settings.SMTP_USERNAME and settings.SMTP_PASSWORD:
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        
        # 发送邮件
        server.send_message(msg)
        
        # 尝试正常关闭连接，但忽略关闭时的错误
        try:
            server.quit()
        except:
            pass
        
        print(f"✅ 验证码已成功发送至 {recipient}")
        return True
    except Exception as e:
        print(f"发送邮件失败: {e}")
        return False

def send_verification_code(email: str, code_type: str) -> bool:
    """发送验证码邮件"""
    # 生成验证码
    code = generate_verification_code()
    
    # 保存验证码到内存
    save_verification_code(email, code, code_type)
    
    # 根据类型确定邮件主题和内容
    if code_type == "register":
        subject = "上海市宜居性系统 - 注册验证码"
        body = f"""
        <html>
        <body>
            <h2>上海市宜居性系统 - 注册验证码</h2>
            <p>您的注册验证码是: <strong>{code}</strong></p>
            <p>该验证码有效期为10分钟。</p>
            <p>如果这不是您发起的请求，请忽略此邮件。</p>
        </body>
        </html>
        """
    elif code_type == "reset_password":
        subject = "上海市宜居性系统 - 密码重置验证码"
        body = f"""
        <html>
        <body>
            <h2>上海市宜居性系统 - 密码重置验证码</h2>
            <p>您的密码重置验证码是: <strong>{code}</strong></p>
            <p>该验证码有效期为10分钟。</p>
            <p>如果这不是您发起的请求，请忽略此邮件并确保您的账户安全。</p>
        </body>
        </html>
        """
    else:
        return False
    
    # 发送邮件
    return send_email(email, subject, body) 
