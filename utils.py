# utils.py
# Comment: Chỉ ra tên file là utils.py (viết tắt của utilities - tiện ích).

import numpy as np
# Import thư viện NumPy, đặt bí danh là `np`. NumPy rất cần thiết cho các phép toán
# trên mảng số học, thường được dùng để xử lý dữ liệu, bao gồm cả việc thiết lập 
# seed ngẫu nhiên và xử lý các frame ảnh trong hàm tạo GIF.

import os
# Import module `os`. Module này cung cấp các hàm để tương tác với hệ điều hành,
# chẳng hạn như tạo thư mục (`os.makedirs`), kiểm tra đường dẫn, v.v. 
# Nó được dùng trong `create_dirs` và `display_frames_as_gif`.

import torch
# Import thư viện PyTorch, một framework học sâu (deep learning) phổ biến.
# Trong file này, nó được sử dụng chủ yếu trong hàm `set_seed` để đảm bảo
# tính lặp lại (reproducibility) của các phép toán ngẫu nhiên trong PyTorch,
# điều rất quan trọng khi huấn luyện mô hình học tăng cường (ví dụ: khởi tạo trọng số mạng nơ-ron).

import matplotlib.pyplot as plt
# Import module `pyplot` từ thư viện Matplotlib và đặt bí danh là `plt`.
# Matplotlib là thư viện vẽ đồ thị phổ biến nhất trong Python. `pyplot` cung cấp
# một giao diện tương tự MATLAB để tạo các loại đồ thị khác nhau. 
# Nó được dùng trong hàm `plot_rewards` để vẽ đồ thị phần thưởng huấn luyện.

import random
# Import module `random` chuẩn của Python. Module này cung cấp các hàm để
# tạo số ngẫu nhiên. Nó được dùng trong `set_seed` để kiểm soát tính ngẫu nhiên
# cơ bản của Python.

from datetime import datetime
# Import lớp `datetime` từ module `datetime`. Lớp này dùng để làm việc với
# ngày và giờ. Nó được sử dụng trong hàm `get_timestamp` để lấy thời gian hiện tại.

def set_seed(seed):
# Định nghĩa một hàm tên là `set_seed` nhận một tham số đầu vào là `seed` (thường là một số nguyên).
# Mục đích của hàm này là thiết lập "hạt giống" (seed) cho các trình tạo số ngẫu nhiên khác nhau.
# Việc này giúp đảm bảo rằng nếu bạn chạy lại code với cùng một giá trị `seed`, 
# chuỗi các số ngẫu nhiên được tạo ra sẽ giống hệt nhau, dẫn đến kết quả thí nghiệm
# (ví dụ: quá trình huấn luyện mô hình) có thể lặp lại được.

    """Đặt seed cho tất cả các nguồn ngẫu nhiên"""
    # Docstring: Mô tả ngắn gọn mục đích của hàm.

    random.seed(seed)
    # Thiết lập seed cho module `random` chuẩn của Python.

    np.random.seed(seed)
    # Thiết lập seed cho trình tạo số ngẫu nhiên của NumPy (`np.random`).

    torch.manual_seed(seed)
    # Thiết lập seed cho PyTorch trên CPU. Điều này ảnh hưởng đến các hoạt động như
    # khởi tạo trọng số mạng nơ-ron, dropout, v.v., khi thực hiện trên CPU.

    if torch.cuda.is_available():
    # Kiểm tra xem máy tính có GPU hỗ trợ CUDA và PyTorch có thể sử dụng nó không.

        torch.cuda.manual_seed(seed)
        # Nếu có GPU, thiết lập seed cho GPU *hiện tại* mà PyTorch đang sử dụng.

        torch.cuda.manual_seed_all(seed)
        # Nếu có GPU, thiết lập seed cho *tất cả* các GPU có sẵn. Điều này quan trọng
        # trong môi trường có nhiều GPU.
        # Lưu ý: Ngay cả khi đặt seed, một số thuật toán trên GPU có thể không hoàn toàn
        # xác định (non-deterministic), nhưng việc đặt seed là bước thực hành tốt nhất.

def create_dirs(dirs):
# Định nghĩa hàm `create_dirs` nhận một tham số `dirs`, dự kiến là một danh sách 
# (list) hoặc một iterable chứa các đường dẫn thư mục dưới dạng chuỗi (string).
# Mục đích: Tạo các thư mục được chỉ định nếu chúng chưa tồn tại.

    """Tạo các thư mục nếu chưa tồn tại"""
    # Docstring mô tả hàm.

    for d in dirs:
    # Bắt đầu một vòng lặp, duyệt qua từng đường dẫn thư mục `d` trong danh sách `dirs`.

        os.makedirs(d, exist_ok=True)
        # Sử dụng hàm `os.makedirs()` để tạo thư mục.
        # `d`: Đường dẫn thư mục cần tạo. `os.makedirs` có khả năng tạo cả các 
        # thư mục cha trung gian nếu chúng chưa tồn tại (ví dụ: tạo 'a/b/c' ngay cả khi 'a' và 'a/b' chưa có).
        # `exist_ok=True`: Đây là một tùy chọn quan trọng. Nếu thư mục `d` đã tồn tại,
        # lệnh này sẽ không báo lỗi và tiếp tục chạy. Nếu không có `exist_ok=True` (hoặc đặt là `False`),
        # nó sẽ gây ra lỗi nếu thư mục đã tồn tại.

def get_timestamp():
# Định nghĩa hàm `get_timestamp` không nhận tham số nào.
# Mục đích: Trả về một chuỗi biểu diễn thời gian hiện tại dưới định dạng cụ thể.

    """Lấy timestamp hiện tại"""
    # Docstring mô tả hàm.

    return datetime.now().strftime("%Y%m%d_%H%M%S")
    # `datetime.now()`: Lấy đối tượng `datetime` đại diện cho ngày và giờ hiện tại.
    # `.strftime("%Y%m%d_%H%M%S")`: Định dạng đối tượng `datetime` thành một chuỗi (string).
    #   - `%Y`: Năm đầy đủ (4 chữ số)
    #   - `%m`: Tháng (01-12)
    #   - `%d`: Ngày trong tháng (01-31)
    #   - `_`: Dấu gạch dưới phân cách ngày và giờ.
    #   - `%H`: Giờ (00-23)
    #   - `%M`: Phút (00-59)
    #   - `%S`: Giây (00-59)
    # Kết quả là một chuỗi như "20231027_103055", rất hữu ích để tạo tên file hoặc 
    # thư mục duy nhất cho mỗi lần chạy thí nghiệm.

def plot_rewards(episode_rewards, avg_rewards, window_size=100, filename='rewards_plot.png'):
# Định nghĩa hàm `plot_rewards` để vẽ đồ thị phần thưởng.
# Tham số:
#   - `episode_rewards`: Một list hoặc array chứa phần thưởng tổng cộng của từng episode.
#   - `avg_rewards`: Một list hoặc array chứa phần thưởng trung bình trượt (moving average) qua các episode.
#   - `window_size=100`: Kích thước cửa sổ dùng để tính trung bình trượt (tham số tùy chọn, mặc định 100). Dùng để hiển thị trong tiêu đề/chú thích.
#   - `filename='rewards_plot.png'`: Tên file để lưu đồ thị (tham số tùy chọn, mặc định là 'rewards_plot.png').

    """Vẽ đồ thị phần thưởng"""
    # Docstring mô tả hàm.

    plt.figure(figsize=(10, 5))
    # Tạo một cửa sổ đồ thị (figure) mới với kích thước xác định (10 inches rộng, 5 inches cao).

    plt.plot(episode_rewards, alpha=0.5, label='Episode Reward')
    # Vẽ đường đồ thị cho `episode_rewards`.
    # `alpha=0.5`: Đặt độ trong suốt của đường là 0.5 (hơi mờ), giúp dễ nhìn hơn nếu đường này nhiễu.
    # `label='Episode Reward'`: Đặt nhãn cho đường này, sẽ được hiển thị trong chú thích (legend).

    plt.plot(avg_rewards, label=f'Average Reward (window={window_size})')
    # Vẽ đường đồ thị cho `avg_rewards` (trung bình trượt).
    # `label=f'...'`: Sử dụng f-string để tạo nhãn động, hiển thị cả kích thước cửa sổ `window_size`.

    plt.xlabel('Episode')
    # Đặt nhãn cho trục hoành (trục x).

    plt.ylabel('Reward')
    # Đặt nhãn cho trục tung (trục y).

    plt.title(f'Training Rewards over Episodes')
    # Đặt tiêu đề cho toàn bộ đồ thị.

    plt.legend()
    # Hiển thị chú thích (legend) dựa trên các nhãn (`label`) đã đặt trong các lệnh `plt.plot`.

    plt.grid(True)
    # Hiển thị lưới trên đồ thị để dễ đọc giá trị hơn.

    plt.savefig(filename)
    # Lưu đồ thị hiện tại vào file có tên được chỉ định bởi `filename`.

    plt.close()
    # Đóng cửa sổ đồ thị hiện tại. Điều này quan trọng để giải phóng bộ nhớ,
    # đặc biệt nếu bạn gọi hàm này nhiều lần trong một vòng lặp dài.

def display_frames_as_gif(frames, filename='game.gif', fps=30):
# Định nghĩa hàm `display_frames_as_gif`.
# Mục đích: Nhận vào một danh sách các frame (hình ảnh) và lưu chúng thành một file ảnh động GIF.
# Tham số:
#   - `frames`: Một list, mỗi phần tử là một frame (thường là một mảng NumPy đại diện cho ảnh).
#   - `filename='game.gif'`: Tên file GIF đầu ra (tùy chọn, mặc định 'game.gif').
#   - `fps=30`: Số khung hình trên giây (frames per second) cho file GIF (tùy chọn, mặc định 30).

    """Lưu một chuỗi frame dưới dạng GIF"""
    # Docstring mô tả hàm.

    from PIL import Image
    # Import lớp `Image` từ thư viện Pillow (PIL fork). Pillow là thư viện chuẩn
    # để xử lý ảnh trong Python. Lệnh import được đặt *bên trong* hàm này, có thể
    # để tránh lỗi nếu Pillow chưa được cài đặt và hàm này không bao giờ được gọi,
    # hoặc để giữ không gian tên (namespace) toàn cục sạch hơn. Tuy nhiên, thông thường
    # import được đặt ở đầu file. Cần đảm bảo đã cài Pillow (`pip install Pillow`).

    import numpy as np
    # Re-import NumPy (np). Điều này là không cần thiết nếu đã import ở đầu file, nhưng không gây hại.

    print(f"Attempting to save {len(frames)} frames to {filename}")
    # In thông báo cho biết hàm đang bắt đầu lưu GIF, số lượng frame và tên file.

    if len(frames) == 0:
    # Kiểm tra xem danh sách `frames` có rỗng không.
        print("Error: No frames to save!")
        # Nếu rỗng, in thông báo lỗi.
        return
        # và kết thúc hàm ngay lập tức.

    # Đảm bảo thư mục tồn tại
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    # Dòng này đảm bảo thư mục chứa file GIF sẽ được tạo nếu chưa có.
    # `os.path.dirname(filename)`: Lấy phần đường dẫn thư mục từ `filename` (ví dụ: trả về 'results' nếu `filename` là 'results/game.gif').
    # `if os.path.dirname(filename) else '.'`: Nếu `filename` không chứa đường dẫn (ví dụ: chỉ là 'game.gif'), `os.path.dirname` trả về chuỗi rỗng, nên ta dùng '.' (thư mục hiện tại).
    # `os.makedirs(..., exist_ok=True)`: Tạo thư mục (và các thư mục cha nếu cần) mà không báo lỗi nếu nó đã tồn tại.

    try:
    # Bắt đầu khối `try`, dùng để bắt các lỗi có thể xảy ra trong quá trình xử lý và lưu file.

        # Đảm bảo frames có định dạng đúng (uint8)
        frames_np = [np.array(frame).astype(np.uint8) if frame is not None else np.zeros((84, 84, 3), dtype=np.uint8) for frame in frames]
        # Đây là một list comprehension để xử lý danh sách `frames` đầu vào:
        # - `for frame in frames`: Lặp qua từng `frame` trong danh sách `frames`.
        # - `np.array(frame)`: Chuyển đổi `frame` thành một mảng NumPy (nếu chưa phải).
        # - `.astype(np.uint8)`: Chuyển kiểu dữ liệu của mảng thành `uint8` (số nguyên không dấu 8-bit, 0-255). Đây là kiểu dữ liệu mà Pillow thường yêu cầu cho ảnh.
        # - `if frame is not None else np.zeros((84, 84, 3), dtype=np.uint8)`: Xử lý trường hợp `frame` có thể là `None`. Nếu là `None`, tạo một frame màu đen (toàn số 0) với kích thước giả định là 84x84 và 3 kênh màu (RGB). Bạn có thể cần điều chỉnh kích thước này nếu frame gốc khác.
        # Kết quả là `frames_np`, một danh sách các mảng NumPy uint8.

        frames_pil = [Image.fromarray(frame) for frame in frames_np]
        # Lại một list comprehension, lần này để chuyển đổi từng mảng NumPy trong `frames_np`
        # thành đối tượng `Image` của Pillow bằng hàm `Image.fromarray()`.

        # Lưu GIF
        frames_pil[0].save(
        # Gọi phương thức `save()` trên đối tượng `Image` đầu tiên (`frames_pil[0]`).
            filename,
            # Tham số đầu tiên là tên file đích.
            save_all=True,
            # Đặt `save_all=True` để chỉ định rằng chúng ta muốn lưu nhiều frame (cho ảnh động).
            append_images=frames_pil[1:],
            # Cung cấp danh sách các đối tượng `Image` còn lại (`frames_pil[1:]`) để nối vào sau frame đầu tiên.
            duration=1000/fps,
            # Đặt thời lượng hiển thị cho mỗi frame tính bằng mili giây (ms). Ví dụ: 30 fps -> 1000/30 ≈ 33 ms/frame.
            loop=0
            # Đặt `loop=0` để GIF lặp lại vô hạn. Các giá trị khác chỉ định số lần lặp cụ thể.
        )
        print(f"Successfully saved GIF to {filename}")
        # In thông báo thành công nếu không có lỗi xảy ra.

    except Exception as e:
    # Nếu có bất kỳ lỗi nào xảy ra trong khối `try` (ví dụ: Pillow không cài, định dạng frame sai, không có quyền ghi file,...).
        print(f"Error saving GIF: {e}")
        # In thông báo lỗi chung, bao gồm cả mô tả lỗi (`e`).
        import traceback
        # Import module `traceback` (chỉ khi có lỗi).
        traceback.print_exc()
        # In ra chi tiết đầy đủ về dấu vết lỗi (traceback), rất hữu ích để gỡ lỗi (debug).