# main.py
import argparse  # Thư viện để phân tích các đối số dòng lệnh.
import sys       # Thư viện cung cấp quyền truy cập vào các biến và hàm được sử dụng hoặc duy trì bởi trình thông dịch Python.

def parse_args():
    """
    Phân tích các đối số dòng lệnh được cung cấp khi chạy script.
    Điều này cho phép người dùng chỉ định chế độ hoạt động (train hoặc test)
    và các đối số bổ sung cho chế độ đó.
    """
    # Tạo một đối tượng ArgumentParser.
    # description là một mô tả ngắn về chương trình, sẽ được hiển thị khi người dùng yêu cầu trợ giúp (ví dụ: python main.py -h).
    parser = argparse.ArgumentParser(description="Pacman Double DQN")

    # Thêm một đối số tên là '--mode'.
    # type=str: Loại dữ liệu của đối số là chuỗi.
    # choices=["train", "test"]: Giá trị của đối số phải là "train" hoặc "test".
    # default="train": Nếu người dùng không cung cấp đối số này, giá trị mặc định sẽ là "train".
    # help: Mô tả về đối số này, hiển thị khi người dùng yêu cầu trợ giúp.
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Mode to run: 'train' or 'test'")

    # Thêm một đối số tên là '--args'.
    # nargs=argparse.REMAINDER: Tất cả các đối số còn lại trên dòng lệnh sau khi các đối số đã biết được phân tích
    #                           sẽ được thu thập vào một danh sách và gán cho 'args'.
    #                           Điều này hữu ích để truyền các đối số cụ thể cho các script train.py hoặc test.py.
    # help: Mô tả về đối số này.
    parser.add_argument("--args", nargs=argparse.REMAINDER,
                        help="Arguments to pass to the selected mode (e.g., --env_name Pacman-v0 --learning_rate 0.001)")

    # Phân tích các đối số từ dòng lệnh (sys.argv[1:]) và trả về một đối tượng Namespace chứa các đối số.
    return parser.parse_args()


def main():
    """
    Hàm chính, điểm vào của chương trình.
    Hàm này sẽ phân tích các đối số dòng lệnh và sau đó gọi hàm main
    của script train.py hoặc test.py tương ứng.
    """
    # Gọi hàm parse_args() để lấy các đối số dòng lệnh đã được phân tích.
    args = parse_args()

    # In ra chế độ hoạt động đã chọn.
    print(f"Running in {args.mode} mode")

    # Kiểm tra giá trị của đối số 'mode'.
    if args.mode == "train":
        # Nếu chế độ là "train":
        # Import hàm main từ module train (giả sử có file train.py với hàm main).
        # Việc import ở đây (local import) giúp tránh import không cần thiết nếu chế độ là "test".
        from train import main as train_main

        # Khởi tạo lại sys.argv cho script train.py.
        # sys.argv là một danh sách các đối số dòng lệnh.
        # sys.argv[0] là tên của script đang chạy (ở đây là main.py).
        # args.args là danh sách các đối số bổ sung được thu thập bởi nargs=argparse.REMAINDER.
        # Dòng này xây dựng lại sys.argv để train_main (hoặc test_main) có thể phân tích các đối số
        # của riêng nó bằng cách sử dụng thư viện argparse (hoặc cách khác) như thể nó được gọi trực tiếp.
        # Ví dụ: nếu chạy `python main.py --mode train --num_episodes 1000`,
        # thì sys.argv cho train_main sẽ là `['main.py', '--num_episodes', '1000']`.
        # Tuy nhiên, thường thì script con (train_main) sẽ muốn tên của *chính nó* là sys.argv[0].
        # Do đó, cách tốt hơn có thể là truyền args.args trực tiếp vào hàm train_main nếu hàm đó chấp nhận.
        # Ở đây, nó đang cố gắng mô phỏng việc script con được gọi với các đối số đó.
        # Nếu args.args là None (không có đối số bổ sung), nó sẽ sử dụng một danh sách rỗng.
        original_argv = sys.argv # Lưu trữ sys.argv gốc
        sys.argv = [original_argv[0]] + (args.args if args.args else [])

        # Chạy hàm main của script train.
        train_main()

        sys.argv = original_argv # Khôi phục sys.argv gốc sau khi train_main kết thúc

    elif args.mode == "test":
        # Nếu chế độ là "test":
        # Import hàm main từ module test (giả sử có file test.py với hàm main).
        from test import main as test_main

        # Khởi tạo lại sys.argv cho script test.py, tương tự như trên.
        original_argv = sys.argv # Lưu trữ sys.argv gốc
        sys.argv = [original_argv[0]] + (args.args if args.args else [])

        # Chạy hàm main của script test.
        test_main()

        sys.argv = original_argv # Khôi phục sys.argv gốc

    else:
        # Nếu chế độ không phải "train" cũng không phải "test".
        # (Điều này không nên xảy ra nếu choices=["train", "test"] hoạt động đúng).
        print(f"Unknown mode: {args.mode}")


# Đây là một idiom phổ biến trong Python.
# Khối mã này chỉ thực thi khi script được chạy trực tiếp (không phải khi được import như một module).
if __name__ == "__main__":
    main()

# Cách sử dụng ví dụ từ dòng lệnh:
# 1. Chạy ở chế độ train (mặc định):
#    python main.py
# 2. Chạy ở chế độ train với các đối số bổ sung cho train.py:
#    python main.py --mode train --learning_rate 0.0001 --episodes 50000
# 3. Chạy ở chế độ test với các đối số bổ sung cho test.py:
#    python main.py --mode test --model_path ./saved_models/pacman_dqn.pth --num_games 5