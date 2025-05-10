# config.py
# Comment: Dòng này chỉ đơn giản là một comment, cho biết tên của file là "config.py".
# Nó hữu ích khi xem code trong một trình soạn thảo không hiển thị tên file rõ ràng
# hoặc khi in code ra giấy.

import ale_py
# Dòng này import thư viện `ale_py`.
# - Mục đích: `ale_py` là một gói Python cung cấp các ràng buộc (bindings) cho Arcade Learning Environment (ALE).
#   ALE là một nền tảng phổ biến để đánh giá các thuật toán học tăng cường trên các trò chơi Atari 2600.
# - Sự cần thiết: Gymnasium (trước đây là OpenAI Gym) sử dụng `ale_py` như một backend để có thể chạy các môi trường
#   game Atari. Khi bạn định nghĩa `ENV_NAME` là một game Atari (ví dụ: "ale_py:ALE/MsPacman-v5"),
#   Gymnasium sẽ tìm và sử dụng `ale_py`. Việc import này ở đầu file cấu hình đảm bảo rằng
#   thư viện này được nhận diện và sẵn sàng khi môi trường được khởi tạo, hoặc đôi khi để kiểm tra
#   xem thư viện đã được cài đặt đúng cách chưa.

"""
Cấu hình cho dự án Pacman Double DQN
"""
# Đây là một "docstring" ở cấp độ module (file).
# - Mục đích: Cung cấp một mô tả ngắn gọn về mục đích hoặc nội dung của toàn bộ file.
#   Trong trường hợp này, nó cho biết file chứa các tham số cấu hình (configuration parameters)
#   cho một dự án huấn luyện agent chơi game Pacman (hoặc MsPacman) sử dụng thuật toán Double DQN.
# - Cách sử dụng: Docstring có thể được truy cập bằng `your_module_name.__doc__` và thường được
#   các công cụ tạo tài liệu tự động (như Sphinx) sử dụng.

class Config:
# Khai báo một lớp (class) tên là `Config`.
# - Mục đích: Lớp này được sử dụng như một "namespace" hoặc một "container" để nhóm tất cả
#   các tham số cấu hình liên quan đến quá trình *huấn luyện* (training) của agent.
#   Việc sử dụng một lớp giúp tổ chức code tốt hơn, dễ dàng truy cập các tham số
#   (ví dụ: `Config.LEARNING_RATE`, `Config.ENV_NAME`) và truyền toàn bộ cấu hình
#   như một đối tượng duy nhất nếu cần.

    # Cấu hình môi trường
    # Comment: Đây là một comment nhóm, cho biết các tham số tiếp theo liên quan đến
    # việc cấu hình môi trường game (environment) mà agent sẽ tương tác.

    ENV_NAME = "ale_py:ALE/MsPacman-v5"  # Thay đổi từ "Pacman-v0"
    # Tên định danh (ID) của môi trường sẽ được sử dụng từ thư viện Gymnasium.
    # - `"ale_py:ALE/MsPacman-v5"`:
    #   - `ale_py:`: Tiền tố chỉ định rằng môi trường này được cung cấp thông qua `ale_py` (Arcade Learning Environment).
    #   - `ALE/`: Một namespace con trong `ale_py`, thường chứa các game Atari.
    #   - `MsPacman-v5`: Tên cụ thể của game là "MsPacman", và `-v5` là phiên bản của API môi trường trong Gymnasium.
    #     MsPacman là một biến thể phổ biến của Pacman, thường được sử dụng trong nghiên cứu học tăng cường.
    # - `# Thay đổi từ "Pacman-v0"`: Ghi chú này cho biết rằng có thể trước đó, dự án đã sử dụng
    #   một môi trường khác có tên "Pacman-v0" (có thể là một phiên bản Pacman từ một nguồn khác hoặc một phiên bản cũ hơn)
    #   và giờ đây đã được cập nhật để sử dụng `MsPacman-v5` từ ALE.

    STACK_FRAMES = 4                    # Số frame để stack
    # Số lượng khung hình (frames) liên tiếp từ môi trường sẽ được "xếp chồng" (stack) lại với nhau
    # để tạo thành một trạng thái (state) duy nhất mà agent nhận được.
    # - Mục đích: Việc xếp chồng frame giúp agent có thể cảm nhận được thông tin về chuyển động
    #   (ví dụ: hướng di chuyển của Ms. Pacman, ma). Nếu chỉ dùng một frame, agent sẽ không biết đối tượng đang đi đâu.
    # - Giá trị `4` là một giá trị phổ biến, được sử dụng trong nhiều bài báo nghiên cứu về DQN, bao gồm cả bài báo gốc.
    # - Tham số này thường được sử dụng bởi một lớp wrapper môi trường (ví dụ: `FrameStack` trong `env.py`).

    FRAME_SIZE = (84, 84)               # Kích thước frame sau khi resize
    # Kích thước (chiều cao, chiều rộng) mong muốn của mỗi frame sau khi được tiền xử lý.
    # - Tiền xử lý thường bao gồm:
    #   1. Chuyển frame màu sang ảnh xám (grayscale) để giảm số chiều dữ liệu.
    #   2. Thay đổi kích thước (resize) frame về một kích thước nhỏ hơn (ví dụ: 84x84) để giảm tải tính toán cho mạng nơ-ron.
    # - Kích thước `(84, 84)` cũng là một tiêu chuẩn được đề xuất trong bài báo DQN gốc của DeepMind.
    # - Tham số này thường được sử dụng bởi một lớp wrapper môi trường (ví dụ: `FrameProcessor` trong `env.py`).

    # Cấu hình agent
    # Comment: Nhóm các tham số liên quan đến bản thân agent và thuật toán học (Double DQN).

    LEARNING_RATE = 2.5e-4              # Tăng tốc độ học
    # Tốc độ học (learning rate) cho thuật toán tối ưu hóa (optimizer, ví dụ: Adam, RMSprop)
    # được sử dụng để cập nhật trọng số của mạng nơ-ron (Q-network).
    # - `2.5e-4` (tương đương 0.00025) là một giá trị tốc độ học khá phổ biến và được coi là hiệu quả
    #   cho các thuật toán DQN trên các game Atari.
    # - `# Tăng tốc độ học`: Ghi chú này có thể ám chỉ rằng giá trị này (2.5e-4) cao hơn một giá trị
    #   đã được sử dụng trước đó trong dự án (ví dụ, có thể trước đó là 1e-4 hoặc 5e-5).
    #   Tốc độ học cao hơn có thể giúp hội tụ nhanh hơn, nhưng cũng có nguy cơ "vọt lố" (overshoot) điểm tối ưu
    #   hoặc làm cho quá trình học không ổn định.

    GAMMA = 0.99                        # Hệ số discount
    # Hệ số chiết khấu (discount factor), ký hiệu là gamma (γ) trong phương trình Bellman của học tăng cường.
    # - Mục đích: Xác định tầm quan trọng của các phần thưởng trong tương lai so với phần thưởng tức thì.
    #   Giá trị `GAMMA` nằm trong khoảng [0, 1].
    # - `0.99`: Một giá trị gần 1 (như 0.99) có nghĩa là agent đánh giá cao các phần thưởng dài hạn.
    #   Điều này phù hợp với các tác vụ mà phần thưởng có thể bị trì hoãn (ví dụ: ăn một "power pellet"
    #   để sau đó ăn được nhiều ma).

    EPSILON_START = 1.0                 # Epsilon ban đầu
    # Giá trị khởi đầu của epsilon (ε) trong chiến lược khám phá epsilon-greedy.
    # - Epsilon-greedy là một chiến lược chọn hành động: với xác suất epsilon, agent chọn một hành động ngẫu nhiên (khám phá);
    #   với xác suất 1-epsilon, agent chọn hành động tốt nhất dựa trên Q-values hiện tại (khai thác).
    # - `1.0`: Nghĩa là ban đầu, agent sẽ hoàn toàn khám phá (chọn hành động ngẫu nhiên 100% thời gian).
    #   Điều này quan trọng để agent thu thập đa dạng kinh nghiệm ban đầu.

    EPSILON_FINAL = 0.1                 # Tăng epsilon cuối cùng
    # Giá trị cuối cùng (nhỏ nhất) mà epsilon sẽ giảm xuống sau một quá trình "decay" (suy giảm).
    # - Mục đích: Ngay cả khi agent đã học được nhiều, việc duy trì một mức độ khám phá nhỏ
    #   (ở đây là 10% với `EPSILON_FINAL = 0.1`) giúp agent có thể thoát khỏi các chính sách dưới tối ưu
    #   và tiếp tục tìm kiếm các giải pháp tốt hơn.
    # - `# Tăng epsilon cuối cùng`: Ghi chú này cho biết giá trị `0.1` này cao hơn giá trị có thể đã được sử dụng
    #   trước đó (ví dụ, 0.05 hoặc 0.01). Điều này có nghĩa là agent sẽ duy trì một mức độ khám phá
    #   cao hơn một chút trong suốt quá trình huấn luyện về sau.

    EPSILON_DECAY = 50000               # Giảm tốc độ decay (huấn luyện nhanh hơn)
    # Số lượng bước (steps) mà trong đó epsilon sẽ giảm dần (decay) từ `EPSILON_START` xuống `EPSILON_FINAL`.
    # - Cách thức decay: Thường là tuyến tính hoặc theo hàm mũ.
    # - `# Giảm tốc độ decay (huấn luyện nhanh hơn)`: Ghi chú này có thể hơi khó hiểu.
    #   - Nếu "tốc độ decay" được hiểu là "thời gian để decay", thì việc "giảm tốc độ decay" nghĩa là làm cho
    #     quá trình decay diễn ra *chậm hơn* (cần nhiều step hơn).
    #   - Tuy nhiên, dựa trên "huấn luyện nhanh hơn", có vẻ ý muốn nói là "giảm *số bước cần thiết* để decay",
    #     tức là epsilon giảm *nhanh hơn*. Với `EPSILON_DECAY = 50000`, epsilon sẽ đạt đến giá trị gần final
    #     sau 50,000 bước. Nếu trước đó giá trị này lớn hơn (ví dụ: 1,000,000), thì việc giảm xuống 50,000
    #     làm cho agent chuyển từ giai đoạn khám phá sang khai thác nhanh hơn.
    #     Điều này có thể làm cho agent *có vẻ* học nhanh hơn ban đầu về mặt điểm số, nhưng có thể bỏ lỡ
    #     việc khám phá đủ không gian trạng thái-hành động.
    #   Chính xác hơn nên hiểu là: Epsilon sẽ giảm từ `EPSILON_START` xuống `EPSILON_FINAL` trong khoảng `50000` bước.
    #   Nếu giá trị này nhỏ, epsilon giảm nhanh. Nếu lớn, epsilon giảm chậm.

    # Cấu hình replay buffer
    # Comment: Nhóm các tham số liên quan đến bộ nhớ đệm kinh nghiệm (Experience Replay Buffer).
    # Replay buffer lưu trữ các tuple kinh nghiệm `(state, action, reward, next_state, done)`
    # và agent lấy mẫu (sample) từ buffer này để huấn luyện, giúp phá vỡ sự tương quan giữa các mẫu liên tiếp.

    BUFFER_SIZE = 50000                 # Giảm kích thước buffer
    # Kích thước tối đa (sức chứa) của replay buffer, tính bằng số lượng tuple kinh nghiệm.
    # - `# Giảm kích thước buffer`: Ghi chú này cho biết kích thước buffer hiện tại (50,000)
    #   nhỏ hơn một giá trị có thể đã được sử dụng trước đó (ví dụ: 100,000, 200,000 hoặc thậm chí 1,000,000
    #   như trong bài báo DQN gốc).
    # - Ảnh hưởng:
    #   - Buffer nhỏ hơn: Tiết kiệm bộ nhớ RAM. Tuy nhiên, agent có thể "quên" các kinh nghiệm cũ nhanh hơn,
    #     và các mẫu trong buffer có thể ít đa dạng hơn, có khả năng dẫn đến "catastrophic forgetting"
    #     hoặc học không ổn định nếu các kinh nghiệm gần đây không đại diện tốt cho toàn bộ môi trường.
    #   - Buffer lớn hơn: Cung cấp nhiều kinh nghiệm đa dạng hơn, giúp ổn định việc học, nhưng tốn nhiều RAM hơn.

    BATCH_SIZE = 32                     # Kích thước batch
    # Số lượng mẫu kinh nghiệm được lấy ngẫu nhiên (hoặc theo độ ưu tiên nếu dùng Prioritized Experience Replay)
    # từ replay buffer trong mỗi bước cập nhật mạng nơ-ron.
    # - `32` là một kích thước batch rất phổ biến và thường được sử dụng cho các thuật toán DQN và biến thể.
    #   Nó là sự cân bằng giữa việc có đủ mẫu để ước lượng gradient tốt và không quá lớn để gây tốn bộ nhớ/tính toán.

    # Cấu hình huấn luyện
    # Comment: Nhóm các tham số liên quan đến vòng lặp huấn luyện chính của agent.

    TARGET_UPDATE = 500              # Giảm số bước giữa mỗi lần cập nhật target network
    # Tần suất (tính bằng số bước huấn luyện của policy network) mà trọng số của mạng đích (target network)
    # sẽ được cập nhật.
    # - Trong DQN và Double DQN, mạng target được sử dụng để tạo ra các giá trị Q mục tiêu ổn định hơn.
    #   Nó được cập nhật định kỳ bằng cách sao chép trọng số từ mạng chính (online/policy network).
    # - `# Giảm số bước...`: Giá trị `500` có nghĩa là mạng target sẽ được cập nhật sau mỗi 500 bước
    #   huấn luyện của mạng policy. Đây là một tần suất cập nhật tương đối thường xuyên so với các giá trị
    #   thường thấy khác (ví dụ: 1000, 10000).
    # - Ảnh hưởng:
    #   - Cập nhật thường xuyên hơn (giá trị `TARGET_UPDATE` nhỏ): Có thể giúp target network bám sát policy network hơn,
    #     có khả năng tăng tốc độ học ban đầu, nhưng cũng có thể làm giảm sự ổn định mà target network mang lại.
    #   - Cập nhật ít thường xuyên hơn (giá trị `TARGET_UPDATE` lớn): Giúp target ổn định hơn, thường dẫn đến việc học
    #     ổn định hơn nhưng có thể chậm hơn.

    TRAIN_STEPS = 100000               # Giảm tổng số bước huấn luyện
    # Tổng số bước tương tác với môi trường (hoặc số lần cập nhật mạng) sẽ được thực hiện trong toàn bộ
    # quá trình huấn luyện.
    # - `# Giảm tổng số bước...`: `100,000` (một trăm nghìn) bước là một số lượng tương đối nhỏ để huấn luyện
    #   các agent DQN trên các game Atari phức tạp (thường cần hàng triệu, thậm chí hàng chục triệu bước
    #   để đạt hiệu năng tốt).
    # - Lý do: Có thể nhằm mục đích chạy thử nghiệm nhanh, kiểm tra thiết lập code, hoặc do giới hạn về
    #   thời gian/tài nguyên tính toán.

    EVAL_INTERVAL = 5000             # Giảm khoảng cách giữa các lần đánh giá
    # Tần suất (tính bằng số bước huấn luyện) mà hiệu năng của agent sẽ được đánh giá (evaluation).
    # - Trong quá trình đánh giá, agent thường được chạy trong một số episode mà không có hành vi khám phá
    #   (ví dụ: epsilon được đặt về 0 hoặc một giá trị rất nhỏ) để đo lường hiệu năng thực tế của chính sách đã học.
    # - `# Giảm khoảng cách...`: Đánh giá sau mỗi 5,000 bước huấn luyện có nghĩa là việc theo dõi tiến trình học
    #   sẽ diễn ra thường xuyên hơn. Điều này hữu ích để vẽ đồ thị học và phát hiện sớm các vấn đề.

    EVAL_EPISODES = 5                   # Số episode để đánh giá
    # Số lượng episode (lượt chơi hoàn chỉnh, từ đầu đến khi game over hoặc đạt mục tiêu) sẽ được chạy
    # trong mỗi lần đánh giá để tính toán một chỉ số hiệu năng (thường là tổng phần thưởng trung bình).
    # - `5` episodes là một con số tương đối nhỏ, cho phép đánh giá nhanh. Để có ước lượng hiệu năng
    #   ổn định hơn và ít nhiễu hơn, số lượng episode đánh giá lớn hơn (ví dụ: 30, 50, 100) sẽ tốt hơn,
    #   nhưng tốn thời gian hơn.

    # Cấu hình lưu trữ
    # Comment: Nhóm các tham số liên quan đến việc lưu trữ dữ liệu, như model checkpoints và logs.

    SAVE_DIR = "checkpoints"            # Thư mục lưu model
    # Tên của thư mục nơi các file checkpoint (chứa trọng số của mô hình đã huấn luyện tại các thời điểm khác nhau)
    # sẽ được lưu lại. Điều này cho phép tiếp tục huấn luyện từ một điểm đã lưu hoặc sử dụng model đã huấn luyện để kiểm thử.

    LOG_DIR = "logs"                    # Thư mục log
    # Tên của thư mục nơi các file log (ví dụ: dữ liệu về phần thưởng qua các episode, giá trị loss,
    # hoặc các dữ liệu khác để theo dõi bằng công cụ như TensorBoard) sẽ được lưu lại.

    SEED = 42                           # Seed cho random
    # Hạt giống (seed) được sử dụng cho các bộ tạo số ngẫu nhiên (random number generators - RNGs)
    # trong các thư viện như `random` của Python, `numpy.random`, và `torch`.
    # - Mục đích: Đặt một seed cố định giúp đảm bảo rằng các kết quả ngẫu nhiên (ví dụ: khởi tạo trọng số mạng,
    #   chọn hành động ngẫu nhiên, lấy mẫu từ buffer) sẽ giống hệt nhau qua các lần chạy khác nhau của chương trình,
    #   miễn là các yếu tố khác không đổi. Điều này rất quan trọng để đảm bảo tính tái lập (reproducibility)
    #   của các thử nghiệm.
    # - `42` là một giá trị seed phổ biến, thường được sử dụng theo quy ước vui (liên quan đến "The Hitchhiker's Guide to the Galaxy").

class TestConfig:
# Khai báo một lớp tên là `TestConfig`.
# - Mục đích: Tương tự như `Config`, lớp này nhóm các tham số cấu hình, nhưng cụ thể là cho quá trình
#   *kiểm thử* (testing) hoặc *chạy thử nghiệm* (inference/evaluation) một mô hình đã được huấn luyện trước đó.

    # Cấu hình kiểm thử
    # Comment: Nhóm các tham số dành riêng cho việc kiểm thử agent.

    MODEL_PATH = None                   # Đường dẫn đến model đã huấn luyện
    # Đường dẫn (path) đến file chứa trọng số của mô hình đã được huấn luyện mà bạn muốn tải (load)
    # để thực hiện kiểm thử.
    # - `None`: Giá trị mặc định này có nghĩa là đường dẫn chưa được chỉ định. Nó thường sẽ được cung cấp
    #   bởi người dùng khi chạy script kiểm thử, ví dụ thông qua một tham số dòng lệnh (command-line argument).

    TEST_EPISODES = 10                  # Số episode để kiểm thử
    # Số lượng episode sẽ được chạy trong quá trình kiểm thử để đánh giá hiệu năng của mô hình đã tải.
    # Tương tự `EVAL_EPISODES`, số lượng lớn hơn sẽ cho kết quả đáng tin cậy hơn.

    RENDER = True                       # Bật render môi trường
    # Một cờ boolean (True/False) xác định liệu có hiển thị giao diện đồ họa của môi trường game
    # trong quá trình kiểm thử hay không.
    # - `True`: Môi trường sẽ được render, cho phép người dùng quan sát trực tiếp cách agent chơi.
    # - `False`: Môi trường sẽ chạy mà không hiển thị, thường nhanh hơn và phù hợp cho việc chạy tự động.

    RECORD = False                      # Có ghi lại video không
    # Một cờ boolean xác định liệu quá trình chơi game của agent trong khi kiểm thử có được ghi lại
    # thành một file video (hoặc GIF) hay không.
    # - `False`: Không ghi lại video.
    # - `True`: Sẽ ghi lại video. Thường cần các thư viện phụ trợ và môi trường phải hỗ trợ.

    VIDEO_DIR = "videos"                # Thư mục lưu video
    # Tên của thư mục nơi các file video/GIF được ghi lại (nếu `RECORD` được đặt là `True`) sẽ được lưu.