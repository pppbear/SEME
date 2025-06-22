// 统一管理项目用户相关的接口
import request from "@/utils/request";
import type { UserLogin, UserSignup, LoginResponse, RegisterResponse, CodeResponse, PasswordReset } from "./type";

// 统一管理接口
enum API {
    LOGIN_URL = "/api/v1/auth/login",
    REGISTER_URL = "/api/v1/auth/register",
    SEND_CODE_URL = "/api/v1/auth/register/send-code",
    RESET_PASSWORD_URL = "/api/v1/auth/reset-password",
    RESET_CODE_URL = "/api/v1/auth/reset-password/send-code"
}

// 暴露请求函数
// 登录接口方法
export const reqLogin = (data: UserLogin): Promise<LoginResponse> => {
  return request.post<LoginResponse>(API.LOGIN_URL, data) as unknown as Promise<LoginResponse>
};

export const reqRegister = (data: UserSignup): Promise<RegisterResponse> => {
  return request.post<RegisterResponse>(API.REGISTER_URL, data) as unknown as Promise<RegisterResponse>
}

export const sendCodeRequest = (data: {email: string}): Promise<CodeResponse> => {
  return request.post<CodeResponse>(API.SEND_CODE_URL, data) as unknown as Promise<CodeResponse>
}

export const passwordReset = (data: PasswordReset): Promise<CodeResponse> => {
  return request.post<PasswordReset>(API.RESET_PASSWORD_URL, data) as unknown as Promise<CodeResponse>
}

export const resetSendCode = (data: {email: string}): Promise<CodeResponse> => {
  return request.post<CodeResponse>(API.RESET_CODE_URL, data) as unknown as Promise<CodeResponse>
}