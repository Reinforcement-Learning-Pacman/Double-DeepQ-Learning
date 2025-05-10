# main.py
# Comment: Tên file là "main.py". File này đóng vai trò là điểm vào (entry point) chính
# của toàn bộ dự án. Nó cho phép người dùng chọn chạy ở chế độ huấn luyện (train)
# hoặc kiểm thử (test) và truyền các tham số cụ thể cho từng chế độ.

import argparse  # Thư viện chuẩn của Python để phân tích các đối số dòng lệnh (command-line arguments).
                 # Giúp tạo các giao diện dòng lệnh thân thiện với người dùng.
import sys       # Thư viện chuẩn của Python cung cấp quyền truy cập vào các biến và hàm
                 # được sử dụng hoặc duy trì bởi trình thông dịch Python, bao gồm `sys.argv` (danh sách các đối số dòng lệnh).

def parse_args():
    """
    Hàm này định nghĩa và phân tích các đối số dòng lệnh được cung cấp khi chạy script `main.py`.
    Trả về một đối tượng `argparse.Namespace` chứa các giá trị của các đối số đã được parse.
    """
    # Tạo một đối tượng `ArgumentParser`.
    # `description` là một mô tả ngắn gọn về mục đích của chương trình,
    # sẽ được hiển thị khi người dùng yêu cầu trợ giúp (ví dụ: `python main.py --help`).
    parser = argparse.ArgumentParser(description="Pacman Double DQN")

    # Thêm một đối số có tên là `--mode`.
    # - `type=str`: Kiểu dữ liệu của đối số này là chuỗi (string).
    # - `choices=["train", "test"]`: Giới hạn các giá trị hợp lệ cho đối số `--mode` là "train" hoặc "test".
    #   Nếu người dùng nhập một giá trị khác, `argparse` sẽ tự động báo lỗi.
    # - `default="train"`: Nếu người dùng không cung cấp đối số `--mode` khi chạy script,
    #   giá trị mặc định của nó sẽ là "train".
    # - `help="Mode to run: 'train' or 'test'"`: Chuỗi mô tả ý nghĩa của đối số này,
    #   sẽ hiển thị trong thông báo trợ giúp.
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Mode to run: 'train' or 'test'")

    # Thêm một đối số có tên là `--args`.
    # - `nargs=argparse.REMAINDER`: Đây là một tùy chọn đặc biệt. Nó có nghĩa là tất cả các đối số
    #   còn lại trên dòng lệnh (sau khi các đối số đã được định nghĩa như `--mode` đã được phân tích)
    #   sẽ được thu thập vào một danh sách và gán cho thuộc tính `args` của đối tượng Namespace trả về.
    #   Dấu `argparse.REMAINDER` thường được sử dụng để "bắt" các đối số không xác định trước
    #   và truyền chúng cho một chương trình con hoặc script khác.
    # - `help="Arguments to pass to the selected mode"`: Chuỗi mô tả cho đối số này.
    parser.add_argument("--args", nargs=argparse.REMAINDER,
                        help="Arguments to pass to the selected mode")

    # Phân tích các đối số từ dòng lệnh (sử dụng `sys.argv[1:]` ngầm định)
    # và trả về một đối tượng `argparse.Namespace`.
    # Ví dụ, nếu chạy `python main.py --mode test --env-name "MyEnv"`,
    # thì `parsed_args.mode` sẽ là "test", và `parsed_args.args` sẽ là `['--env-name', 'MyEnv']`.
    return parser.parse_args()


def main():
    """
    Hàm chính (main entry point) của chương trình.
    Hàm này sẽ được gọi khi script `main.py` được thực thi trực tiếp.
    """
    # Bước 1: Parse các đối số dòng lệnh mà người dùng cung cấp.
    args = parse_args()

    # In ra chế độ hoạt động đã được chọn (train hoặc test) ra console.
    print(f"Running in {args.mode} mode")

    # Bước 2: Kiểm tra giá trị của `args.mode` để quyết định hành động tiếp theo.
    if args.mode == "train":
        # Nếu chế độ được chọn là "train":
        # Import hàm `main` từ module `train` (tức là file `train.py`)
        # và đặt bí danh (alias) cho nó là `train_main`.
        # Việc import được thực hiện bên trong khối `if` này để tránh import không cần thiết
        # nếu chế độ không phải là "train" (lazy import).
        from train import main as train_main

        # Khởi tạo lại `sys.argv` cho script `train.py`.
        # - `sys.argv` là một danh sách các đối số dòng lệnh của Python.
        #   `sys.argv[0]` luôn là tên của script đang chạy (trong trường hợp này là "main.py" hoặc đường dẫn đầy đủ đến nó).
        # - Mục đích của việc này là để script `train.py` (hoặc `test.py`) có thể sử dụng `argparse`
        #   để tự phân tích các đối số của riêng nó, như thể nó được gọi trực tiếp từ dòng lệnh
        #   với các đối số đó.
        # - `[sys.argv[0]]`: Giữ lại tên của script gốc (`main.py`).
        # - `+ (args.args if args.args else [])`: Nối thêm danh sách các đối số được thu thập bởi
        #   `argparse.REMAINDER` (lưu trong `args.args`).
        #   Nếu `args.args` là `None` hoặc rỗng (nghĩa là không có đối số nào được truyền cho chế độ train),
        #   thì sẽ nối một danh sách rỗng (`[]`).
        #   Ví dụ: Nếu `args.args` là `['--lr', '0.001']`, thì `sys.argv` mới sẽ là `['main.py', '--lr', '0.001']`.
        sys.argv = [sys.argv[0]] + (args.args if args.args else [])

        # Gọi hàm `train_main()` (đã import từ `train.py`) để bắt đầu quá trình huấn luyện.
        # Script `train.py` bây giờ sẽ thấy các đối số (nếu có) trong `sys.argv` và có thể parse chúng.
        train_main()
    elif args.mode == "test":
        # Nếu chế độ được chọn là "test":
        # Tương tự như trường hợp "train", import hàm `main` từ module `test` (file `test.py`)
        # và đặt bí danh là `test_main`.
        from test import main as test_main

        # Khởi tạo lại `sys.argv` cho script `test.py` với các đối số được truyền cho chế độ test.
        sys.argv = [sys.argv[0]] + (args.args if args.args else [])

        # Gọi hàm `test_main()` (đã import từ `test.py`) để bắt đầu quá trình kiểm thử.
        test_main()
    else:
        # Nếu giá trị của `args.mode` không phải là "train" hay "test"
        # (Mặc dù `choices` trong `argparse` đã giới hạn điều này, đây là một biện pháp phòng ngừa
        # hoặc cho trường hợp logic có thể thay đổi trong tương lai).
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    # Đây là một cấu trúc chuẩn và rất phổ biến trong các script Python.
    # - `__name__` là một biến đặc biệt (built-in variable) trong Python.
    #   - Khi một script Python được chạy trực tiếp (ví dụ, bằng cách gõ `python main.py` trong terminal),
    #     giá trị của `__name__` bên trong script đó sẽ được đặt là `"__main__"`.
    #   - Nếu script được import như một module vào một script khác (ví dụ: `import main_module`),
    #     thì giá trị của `__name__` bên trong `main_module.py` sẽ là tên của module đó (tức là "main_module").
    # - Do đó, khối code bên trong `if __name__ == "__main__":` chỉ được thực thi khi script này
    #   được chạy trực tiếp từ dòng lệnh. Nó không được thực thi nếu script này được import bởi một script khác.
    #   Điều này làm cho hàm `main()` trở thành điểm khởi đầu thực sự của chương trình khi nó được chạy.
    main()