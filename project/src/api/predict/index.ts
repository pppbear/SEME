// 统一管理项目用户相关的接口
import request from "@/utils/request";

// 统一管理接口
enum API {
    COMPARE_URL = "/api/v1/predict/dependent_predict"
}

export interface PredictResponse {
    code: number
    message: string
    data: Array<number>
}

// 暴露请求函数
export const predictResult = (data: {file: File, dependent_name: string}): Promise<PredictResponse> => {
  return request.post<PredictResponse>(API.COMPARE_URL, data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }) as unknown as Promise<PredictResponse>
};
