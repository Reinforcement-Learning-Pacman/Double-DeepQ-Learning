import gymnasium as gym # Nhập thư viện Gymnasium

# Lấy danh sách tất cả các ID môi trường đã được đăng ký trong Gymnasium.
# gym.envs.registry là một dictionary nơi khóa là ID của môi trường (ví dụ: 'CartPole-v1')
# và giá trị là thông tin về môi trường đó (entry point, etc.).
# .keys() trả về một view object chứa các khóa của dictionary này.
all_envs = gym.envs.registry.keys()

# Lọc ra các môi trường Atari.
# Các môi trường Atari trong Gymnasium thường có ID bắt đầu bằng 'ALE/' (nếu dùng gymnasium[atari])
# hoặc 'ale_py:ALE/' (nếu được đăng ký trực tiếp bởi ale-py, cách này có vẻ đang được dùng ở đây).
# Chúng ta tạo một list comprehension để duyệt qua tất cả các ID môi trường
# và chỉ giữ lại những ID nào bắt đầu bằng chuỗi 'ale_py:ALE/'.
# atari_envs = [env_id for env_id in all_envs if env_id.startswith('ale_py:ALE/')]
# Bạn cũng có thể kiểm tra `env_id.startswith('ALE/')` nếu bạn đang sử dụng
# cách đăng ký môi trường Atari khác của Gymnasium.

# In ra tiêu đề cho danh sách các môi trường Atari.
print("All Atari environments (registered via ale_py):")

# Duyệt qua danh sách các môi trường Atari đã lọc được và in ra từng ID.
# Điều này giúp bạn xác nhận tên chính xác của môi trường Pacman (và các game Atari khác)
# mà bạn có thể sử dụng với `gym.make()`.
for env_id in all_envs:
    print(f" - {env_id}")

# Ghi chú thêm:
# Mục đích của đoạn mã này là để liệt kê các môi trường Atari có sẵn
# trong bản cài đặt Gymnasium của bạn. Điều này hữu ích để:
# 1. Xác nhận rằng các môi trường Atari đã được cài đặt và đăng ký đúng cách.
# 2. Tìm tên ID chính xác của môi trường bạn muốn sử dụng (ví dụ: 'ale_py:ALE/Pacman-v5').
#    Tên này có thể thay đổi một chút giữa các phiên bản Gymnasium hoặc cách cài đặt.
#
# Nếu danh sách này trống hoặc không chứa Pacman, bạn có thể cần cài đặt hoặc
# cấu hình lại phần phụ thuộc Atari cho Gymnasium.
# Thông thường, bạn sẽ cần `pip install gymnasium[atari]` và có thể cả `pip install gymnasium[accept-rom-license]`
# (hoặc `pip install autorom[accept-rom-license]` để tự động tải ROMs).