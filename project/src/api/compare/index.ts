// 统一管理项目用户相关的接口
import request from "@/utils/request";

// 统一管理接口
enum API {
    COMPARE_URL = "/api/v1/compare/model_compare"
}

export interface CompareResult {
    dependent_name: string
    true_values: Array<number>
    mlp_predictions: Array<number>
    rf_predictions: Array<number>
    kan_predictions: Array<number>
    mse_mlp: number
    r2_mlp: number
    mse_rf: number
    r2_rf: number
    mse_kan: number
    r2_kan: number
}

export interface CompareResponse {
    code: number
    message: string
    data: Array<CompareResult>
}

// 暴露请求函数
export const getCompareResult = (data: {file: File, dependent_name: string}): Promise<CompareResponse> => {
  return request.post<CompareResponse>(API.COMPARE_URL, data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  }) as unknown as Promise<CompareResponse>
};
