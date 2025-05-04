import ale_py
# Dòng này import thư viện `ale_py`. 
# `ale_py` là một thư viện Python cung cấp liên kết (bindings) đến 
# Arcade Learning Environment (ALE). ALE là một framework nổi tiếng 
# cho phép các agent học tăng cường (RL agents) tương tác với hàng trăm 
# game Atari 2600. 
# Việc import `ale_py` là cần thiết để Gymnasium có thể nhận biết và 
# tạo ra các môi trường game Atari (ví dụ: "Breakout-v5", "Pong-v5", "MsPacman-v5"...). 
# Khi bạn import `ale_py`, các môi trường Atari tương ứng sẽ được "đăng ký" (register)
# với Gymnasium, làm cho chúng có sẵn để sử dụng thông qua `gym.make()`.

# if using gymnasium
# Đây là một dòng comment (ghi chú), chỉ ra rằng dòng code tiếp theo (đã bị comment)
# có thể liên quan hoặc cần thiết khi sử dụng thư viện Gymnasium.

# import shimmy
# Dòng này bị comment out (có dấu # ở đầu), nghĩa là nó sẽ không được thực thi.
# `shimmy` là một thư viện tiện ích được thiết kế để cung cấp lớp tương thích (compatibility layer).
# Mục đích chính của nó là giúp các môi trường được viết cho API cũ của `gym` (phiên bản trước 0.26)
# có thể hoạt động được với API mới hơn của `gymnasium`. 
# Ví dụ, API mới của `gymnasium` yêu cầu hàm `step()` trả về 5 giá trị 
# (observation, reward, terminated, truncated, info), trong khi API cũ chỉ trả về 4 giá trị 
# (observation, reward, done, info). `shimmy` có thể "bao bọc" (wrap) môi trường cũ 
# để nó hoạt động theo chuẩn mới.
# Trong trường hợp này, nó được comment out, có thể vì các môi trường Atari 
# được cung cấp qua `ale_py` đã tương thích sẵn với `gymnasium`, hoặc 
# đoạn code này chỉ nhằm mục đích liệt kê môi trường chứ chưa cần chạy một môi trường cũ cụ thể.

import gymnasium as gym
# Dòng này import thư viện `gymnasium` và đặt cho nó một bí danh (alias) là `gym`.
# `gymnasium` là thư viện cốt lõi để tạo và tương tác với các môi trường học tăng cường. 
# Việc sử dụng bí danh `gym` là một quy ước phổ biến, giúp viết code ngắn gọn hơn.

print(gym.envs.registry.keys()) 
# Dòng này thực hiện các bước sau:
# 1. `gym`: Truy cập vào thư viện `gymnasium` thông qua bí danh đã đặt.
# 2. `.envs`: Truy cập vào một module hoặc đối tượng bên trong `gymnasium` liên quan đến các môi trường.
# 3. `.registry`: Truy cập vào "registry" (sổ đăng ký) của các môi trường. Registry này là một cấu trúc dữ liệu 
#    (thường giống như một dictionary) lưu trữ thông tin về tất cả các môi trường mà `gymnasium` 
#    (và các plugin như `ale_py` đã được import) biết đến và có thể tạo ra. Mỗi môi trường được định danh 
#    bằng một ID duy nhất (ví dụ: "CartPole-v1", "Pong-v5").
# 4. `.keys()`: Gọi phương thức `.keys()` trên đối tượng registry. Nếu registry là một dictionary, 
#    phương thức này sẽ trả về một danh sách (hoặc một đối tượng view giống danh sách) chứa tất cả 
#    các "key" của dictionary đó. Trong ngữ cảnh này, các "key" chính là các ID của môi trường đã được đăng ký.
# 5. `print(...)`: In danh sách các ID môi trường này ra màn hình console.

# ==> Mục đích cuối cùng của dòng này là hiển thị tất cả các ID của những môi trường 
#     mà bạn có thể tạo bằng lệnh `gym.make("environment_id")` trong phiên làm việc Python hiện tại, 
#     bao gồm cả các môi trường tích hợp sẵn của Gymnasium và các môi trường Atari được thêm vào bởi `ale_py`.