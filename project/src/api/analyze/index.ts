// 统一管理项目用户相关的接口
import request from "@/utils/request";

// 统一管理接口
enum API {
    COMPARE_URL = "/api/v1/analyze/dependent_feature_analyze"
}

interface KeyFeature {
    feature_name: string
    feature_value: number
}

export interface AnalyzeResponse {
    code: number
    message: string
    data: Array<KeyFeature>
}

// 暴露请求函数
export const predictResult = (data: {file: File, dependent_name: string}): Promise<AnalyzeResponse> => {
  return request.post<AnalyzeResponse>(API.COMPARE_URL, data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }) as unknown as Promise<AnalyzeResponse>
};
