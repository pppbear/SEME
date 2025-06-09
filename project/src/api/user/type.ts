export interface UserLogin {
    username: string,
    password: string,
    remember: boolean
}

export interface UserSignup {
    email: string,
    password: string,
    username: string,
    verification_code: string
}

interface LoginData {
  token: string;
  expire: number;
  role: string;
  username: string;
}

export interface LoginResponse {
  code: number;
  data: LoginData;
  message: string;
}

interface RegisterData {
    id: number;
    username: string;
    email: string;
    is_active: boolean;
    created_at: Date;
}

export interface RegisterResponse {
  code: number;
  data: RegisterData;
  message: string;
}

export interface CodeResponse {
  code: number;
  message: string;
}

export interface PasswordReset {
    email: string,
    new_password: string,
    verification_code: string
}