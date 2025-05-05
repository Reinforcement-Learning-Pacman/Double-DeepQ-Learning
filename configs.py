# config.py
# Comment: Cho biết tên của file là config.py.

import ale_py
# Dòng này import thư viện `ale_py`. Như đã giải thích trước đó, `ale_py` cung cấp 
# giao diện để Gymnasium có thể sử dụng các môi trường game Atari từ Arcade Learning Environment (ALE).
# Việc import này cần thiết để dòng `ENV_NAME = "ale_py:ALE/MsPacman-v5"` bên dưới hoạt động đúng.

"""
Cấu hình cho dự án Pacman Double DQN
"""
# Đây là một docstring cấp độ module, mô tả mục đích chung của file: 
# chứa các tham số cấu hình cho dự án huấn luyện agent chơi game Pacman bằng thuật toán Double DQN.

class Config:
# Khai báo một lớp (class) tên là `Config`. 
# Mục đích: Nhóm tất cả các tham số cấu hình liên quan đến quá trình *huấn luyện* (training) 
# agent vào một nơi duy nhất. Việc sử dụng lớp giúp truy cập các cấu hình này một cách có tổ chức 
# (ví dụ: `Config.LEARNING_RATE`, `Config.ENV_NAME`).

    # Cấu hình môi trường
    # Comment: Nhóm các tham số liên quan đến môi trường game.

    ENV_NAME = "ale_py:ALE/MsPacman-v5"  # Thay đổi từ "Pacman-v0"
    # Tên định danh (ID) của môi trường Gymnasium sẽ được sử dụng.
    # - `"ale_py:ALE/MsPacman-v5"`: Chỉ định rõ rằng môi trường này đến từ `ale_py`, 
    #   thuộc nhóm `ALE` (Arcade Learning Environment), và tên cụ thể là `MsPacman-v5`. 
    #   MsPacman là một biến thể phổ biến của Pacman. v5 là phiên bản API của môi trường trong Gymnasium.
    # - `# Thay đổi từ "Pacman-v0"`: Ghi chú này cho biết trước đó có thể đã sử dụng môi trường "Pacman-v0" 
    #   (có thể là một phiên bản khác hoặc môi trường từ nguồn khác) và giờ đã được đổi thành MsPacman từ ALE.

    STACK_FRAMES = 4                    # Số frame để stack
    # Số lượng frame (khung hình) liên tiếp từ môi trường sẽ được xếp chồng (stack) lên nhau 
    # để tạo thành một trạng thái (state) duy nhất cho agent. 
    # Giá trị 4 là phổ biến, giúp agent nhận biết được chuyển động. Tham số này được sử dụng bởi lớp `FrameStack` trong `env.py`.

    FRAME_SIZE = (84, 84)               # Kích thước frame sau khi resize
    # Kích thước (chiều cao, chiều rộng) mong muốn của mỗi frame sau khi được tiền xử lý 
    # (chuyển sang ảnh xám và resize). Kích thước (84, 84) là tiêu chuẩn từ bài báo DQN gốc. 
    # Tham số này được sử dụng bởi lớp `FrameProcessor` trong `env.py`.
    
    # Cấu hình agent
    # Comment: Nhóm các tham số liên quan đến bản thân agent và thuật toán học (Double DQN).

    LEARNING_RATE = 2.5e-4              # Tăng tốc độ học
    # Tốc độ học (learning rate) cho thuật toán tối ưu hóa (ví dụ: Adam, RMSprop) được sử dụng 
    # để cập nhật trọng số của mạng nơ-ron (Q-network). 
    # `2.5e-4` (tức là 0.00025) là một giá trị khá phổ biến cho các thuật toán DQN trên Atari.
    # `# Tăng tốc độ học`: Ghi chú này có thể ám chỉ rằng giá trị này cao hơn một giá trị đã dùng trước đó, 
    # hoặc nó được coi là tương đối "nhanh" so với các giá trị nhỏ hơn (ví dụ: 1e-4).

    GAMMA = 0.99                        # Hệ số discount
    # Hệ số chiết khấu (discount factor) gamma (γ) trong phương trình Bellman. 
    # Nó xác định tầm quan trọng của phần thưởng trong tương lai so với phần thưởng tức thì. 
    # Giá trị gần 1 (như 0.99) có nghĩa là agent quan tâm nhiều đến phần thưởng dài hạn.

    EPSILON_START = 1.0                 # Epsilon ban đầu
    # Giá trị khởi đầu của epsilon (ε) trong chiến lược khám phá epsilon-greedy. 
    # `1.0` nghĩa là ban đầu agent sẽ hoàn toàn khám phá (chọn hành động ngẫu nhiên 100% thời gian).

    EPSILON_FINAL = 0.1                 # Tăng epsilon cuối cùng
    # Giá trị cuối cùng (nhỏ nhất) của epsilon sau khi đã giảm dần (decay). 
    # Agent sẽ luôn giữ lại một xác suất nhỏ (`0.1` tức 10%) để chọn hành động ngẫu nhiên, 
    # nhằm tránh bị kẹt trong chính sách dưới tối ưu.
    # `# Tăng epsilon cuối cùng`: Ghi chú cho biết giá trị này (0.1) cao hơn giá trị trước đó (có thể là 0.05 hoặc 0.01),
    # nghĩa là agent sẽ duy trì mức độ khám phá cao hơn một chút về sau này.

    EPSILON_DECAY = 20000               # Giảm tốc độ decay (huấn luyện nhanh hơn)
    # Số lượng bước (steps) mà trong đó epsilon sẽ giảm dần từ `EPSILON_START` xuống `EPSILON_FINAL`. 
    # `# Giảm tốc độ decay`: Ghi chú này có vẻ hơi ngược. Một giá trị `EPSILON_DECAY` *nhỏ hơn* (như 50000 so với ví dụ 1 triệu) 
    # có nghĩa là epsilon giảm *nhanh hơn*, agent chuyển sang khai thác sớm hơn. Điều này có thể làm quá trình *có vẻ* 
    # hội tụ nhanh hơn về mặt điểm số ban đầu nhưng không chắc đã tốt hơn về mặt khám phá không gian trạng thái. 
    # Nên hiểu là "giảm *số bước* để decay", làm quá trình decay diễn ra nhanh hơn.
    
    # Cấu hình replay buffer
    # Comment: Nhóm các tham số liên quan đến bộ nhớ đệm kinh nghiệm (Experience Replay Buffer).

    BUFFER_SIZE = 10000                 # Giảm kích thước buffer
    # Kích thước tối đa của replay buffer (số lượng tuple kinh nghiệm `(state, action, reward, next_state, done)` có thể lưu trữ).
    # `# Giảm kích thước buffer`: Ghi chú này cho biết kích thước buffer (50k) nhỏ hơn giá trị trước đó (có thể là 100k, 200k hoặc 1 triệu). 
    # Buffer nhỏ hơn tiết kiệm bộ nhớ RAM nhưng có thể khiến agent "quên" các kinh nghiệm cũ nhanh hơn.

    BATCH_SIZE = 32                     # Kích thước batch
    # Số lượng mẫu kinh nghiệm được lấy ngẫu nhiên từ replay buffer trong mỗi bước cập nhật mạng nơ-ron. 
    # `32` là một kích thước batch rất phổ biến cho DQN.
    
    # Cấu hình huấn luyện
    # Comment: Nhóm các tham số liên quan đến vòng lặp huấn luyện chính.

    TARGET_UPDATE = 1000                 # Giảm số bước giữa mỗi lần cập nhật target network
    # Tần suất (số bước huấn luyện) cập nhật trọng số của mạng target (target network). 
    # Trong DQN và Double DQN, mạng target được cập nhật định kỳ bằng cách sao chép trọng số từ mạng chính (online network).
    # `# Giảm số bước...`: Giá trị 500 nhỏ hơn các giá trị thường thấy khác (ví dụ: 1000, 10000). 
    # Cập nhật thường xuyên hơn có thể giúp hội tụ nhanh hơn nhưng cũng có thể gây mất ổn định.

    TRAIN_STEPS = 100000                # Giảm tổng số bước huấn luyện
    # Tổng số bước tương tác với môi trường (training steps) sẽ được thực hiện trong toàn bộ quá trình huấn luyện.
    # `# Giảm tổng số bước...`: 500k là một số lượng bước tương đối nhỏ cho các game Atari phức tạp (thường cần hàng triệu bước). 
    # Điều này có thể nhằm mục đích chạy thử nghiệm nhanh hoặc do giới hạn tài nguyên.

    EVAL_INTERVAL = 10000                # Giảm khoảng cách giữa các lần đánh giá
    # Tần suất (số bước huấn luyện) thực hiện việc đánh giá (evaluation) hiệu năng của agent. 
    # Trong quá trình đánh giá, agent thường chạy mà không có khám phá (epsilon nhỏ hoặc bằng 0) để đo lường hiệu năng thực tế.
    # `# Giảm khoảng cách...`: Đánh giá thường xuyên hơn (mỗi 5k bước) giúp theo dõi tiến trình học tốt hơn.

    EVAL_EPISODES = 3                # Số episode để đánh giá
    # Số lượng episode (lượt chơi hoàn chỉnh) sẽ được chạy trong mỗi lần đánh giá để tính toán điểm số trung bình. 
    # 5 episodes là một con số hợp lý để có ước lượng sơ bộ về hiệu năng.
    
    # Cấu hình lưu trữ
    # Comment: Nhóm các tham số liên quan đến việc lưu trữ dữ liệu.

    SAVE_DIR = "checkpoints"            # Thư mục lưu model
    # Tên thư mục nơi các file checkpoint (trọng số của mô hình đã huấn luyện) sẽ được lưu lại.

    LOG_DIR = "logs"                    # Thư mục log
    # Tên thư mục nơi các file log (ví dụ: dữ liệu phần thưởng, log cho TensorBoard) sẽ được lưu lại.

    SEED = 42                           # Seed cho random
    # Hạt giống (seed) cho các trình tạo số ngẫu nhiên (Python random, NumPy, PyTorch). 
    # Sử dụng seed cố định giúp đảm bảo kết quả huấn luyện có thể lặp lại được. Giá trị 42 là một lựa chọn phổ biến theo quy ước.

class TestConfig:
# Khai báo một lớp tên là `TestConfig`.
# Mục đích: Nhóm các tham số cấu hình liên quan đến quá trình *kiểm thử* (testing) hoặc *chạy thử* 
# (inference) một mô hình đã được huấn luyện.

    # Cấu hình kiểm thử
    # Comment: Nhóm các tham số cho việc kiểm thử.

    MODEL_PATH = None                   # Đường dẫn đến model đã huấn luyện
    # Đường dẫn đến file chứa trọng số của mô hình đã huấn luyện cần được tải để kiểm thử. 
    # `None` là giá trị mặc định, cần được thay đổi thành đường dẫn thực tế trước khi chạy kiểm thử.

    TEST_EPISODES = 10                  # Số episode để kiểm thử
    # Số lượng episode sẽ chạy trong quá trình kiểm thử để đánh giá hiệu năng của mô hình đã tải.

    RENDER = True                       # Bật render môi trường
    # Cờ boolean xác định có hiển thị giao diện đồ họa của môi trường game trong quá trình kiểm thử hay không. 
    # `True` nghĩa là có hiển thị.

    RECORD = False                      # Có ghi lại video không
    # Cờ boolean xác định có ghi lại quá trình chơi game thành file video/GIF hay không. 
    # `False` nghĩa là không ghi lại.

    VIDEO_DIR = "videos"                # Thư mục lưu video 
    # Tên thư mục nơi các file video/GIF được ghi lại (nếu `RECORD` là `True`) sẽ được lưu.