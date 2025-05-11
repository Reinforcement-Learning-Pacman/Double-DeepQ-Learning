import numpy as np  # Thư viện NumPy cho các phép toán số học, ở đây dùng để tính kích thước output của lớp conv.
import torch  # Thư viện PyTorch chính cho deep learning.
import torch.nn as nn  # Module `nn` của PyTorch chứa các lớp để xây dựng mạng neural.


class DQNModel(nn.Module):
    """
    Lớp định nghĩa kiến trúc mạng Deep Q-Network (DQN).
    Mạng này bao gồm các lớp convolutional (CNN) để trích xuất đặc trưng từ ảnh đầu vào (các frame của game)
    và các lớp fully connected (FC) để ánh xạ các đặc trưng đó sang Q-values cho mỗi hành động.
    """
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        """
        Khởi tạo các lớp của mạng DQN.

        Args:
            input_shape (tuple): Hình dạng của tensor đầu vào, thường là (channels, height, width).
                                 Ví dụ: (4, 84, 84) cho 4 frame xếp chồng, mỗi frame 84x84 pixel.
            n_actions (int): Số lượng hành động mà agent có thể thực hiện.
                             Mạng sẽ output một Q-value cho mỗi hành động này.
        """
        super(DQNModel, self).__init__()  # Gọi constructor của lớp cha (nn.Module).

        # --- Convolutional Neural Network (CNN) part ---
        # `nn.Sequential` là một container tuần tự. Các module sẽ được thêm vào nó theo thứ tự chúng được truyền vào constructor.
        # CNN được sử dụng để trích xuất các đặc trưng không gian từ các frame hình ảnh của trò chơi.
        self.conv = nn.Sequential(
            # Lớp Convolutional đầu tiên:
            # input_shape[0]: Số kênh đầu vào (ví dụ: 4 nếu stack 4 frame).
            # 32: Số kênh đầu ra (số lượng feature map).
            # kernel_size=8: Kích thước của bộ lọc (filter) là 8x8.
            # stride=4: Bước nhảy của bộ lọc là 4 pixel.
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),  # Hàm kích hoạt ReLU (Rectified Linear Unit) để thêm tính phi tuyến.

            # Lớp Convolutional thứ hai:
            # 32: Số kênh đầu vào (phải khớp với số kênh đầu ra của lớp trước).
            # 64: Số kênh đầu ra.
            # kernel_size=4: Kích thước bộ lọc 4x4.
            # stride=2: Bước nhảy 2 pixel.
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),

            # Lớp Convolutional thứ ba:
            # 64: Số kênh đầu vào.
            # 64: Số kênh đầu ra.
            # kernel_size=3: Kích thước bộ lọc 3x3.
            # stride=1: Bước nhảy 1 pixel.
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # --- Fully Connected (FC) part / Linear layers ---
        # Các lớp này nhận đầu vào là vector đặc trưng đã được "làm phẳng" (flattened) từ output của CNN
        # và output ra Q-values cho mỗi hành động.
        self.fc = nn.Sequential(
            # Lớp Linear đầu tiên (fully connected):
            # self._get_conv_output(input_shape): Số lượng đặc trưng đầu vào.
            #                                     Đây là kích thước của output từ CNN sau khi được làm phẳng.
            # 512: Số lượng unit (neuron) trong lớp ẩn này.
            nn.Linear(self._get_conv_output(input_shape), 512),
            nn.ReLU(),  # Hàm kích hoạt ReLU.

            # Lớp Linear thứ hai (lớp output):
            # 512: Số lượng unit đầu vào (phải khớp với output của lớp trước).
            # n_actions: Số lượng unit đầu ra, tương ứng với Q-value cho mỗi hành động.
            nn.Linear(512, n_actions)
        )

    def _get_conv_output(self, shape: tuple) -> int:
        """
        Hàm trợ giúp để tính toán kích thước của vector đặc trưng sau khi đi qua các lớp convolutional.
        Điều này cần thiết để xác định số lượng input unit cho lớp fully connected đầu tiên.

        Args:
            shape (tuple): Hình dạng của tensor đầu vào ban đầu (channels, height, width).

        Returns:
            int: Tổng số phần tử trong tensor output của các lớp convolutional (kích thước của vector đặc trưng đã làm phẳng).
        """
        # Tạo một tensor giả (dummy tensor) có cùng hình dạng với batch đầu vào (batch_size=1).
        # *shape giải nén tuple `shape` thành các đối số riêng lẻ (ví dụ: 1, channels, height, width).
        batch = torch.zeros(1, *shape)
        # Cho tensor giả đi qua các lớp convolutional.
        conv_out = self.conv(batch)
        # Tính tổng số phần tử trong tensor output.
        # conv_out.size() trả về một torch.Size object (ví dụ: [1, 64, 7, 7]).
        # np.prod tính tích của tất cả các phần tử trong tuple kích thước đó.
        return int(np.prod(conv_out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Định nghĩa quá trình truyền thẳng (forward pass) của mạng.
        Đây là nơi dữ liệu đầu vào được xử lý qua các lớp của mạng để tạo ra output.

        Args:
            x (torch.Tensor): Tensor đầu vào, thường là một batch các state của game.
                              Kỳ vọng hình dạng: (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Tensor output, chứa Q-values cho mỗi hành động.
                          Hình dạng: (batch_size, n_actions).
        """
        # Chuẩn hóa giá trị pixel của ảnh đầu vào về khoảng [0, 1].
        # Ảnh đầu vào thường có giá trị pixel từ 0 đến 255.
        # Việc chuẩn hóa giúp ổn định quá trình huấn luyện.
        x = x.float() / 255.0

        # Cho đầu vào đi qua các lớp convolutional.
        # conv_out sẽ có hình dạng (batch_size, num_output_channels_conv, out_height, out_width).
        conv_out = self.conv(x)

        # Làm phẳng (flatten) output từ các lớp convolutional.
        # `conv_out.view(x.size()[0], -1)` thay đổi hình dạng của `conv_out`.
        # x.size()[0] là batch_size.
        # -1 tự động tính toán kích thước còn lại để tổng số phần tử không đổi.
        # Kết quả là một tensor 2D có hình dạng (batch_size, num_features),
        # trong đó num_features là tích của các chiều còn lại (channels * height * width) của conv_out.
        conv_out_flat = conv_out.view(x.size()[0], -1)

        # Cho vector đặc trưng đã làm phẳng đi qua các lớp fully connected.
        # Output sẽ là Q-values cho mỗi hành động.
        return self.fc(conv_out_flat)

# Ví dụ sử dụng (tùy chọn, để kiểm tra)
if __name__ == '__main__':
    # Giả sử input là 4 frame 84x84 pixel, và có 5 hành động khả thi.
    input_s = (4, 84, 84)
    num_actions = 5
    model = DQNModel(input_s, num_actions)
    print(model) # In ra kiến trúc của model

    # Tạo một batch dữ liệu đầu vào giả (ví dụ: 2 state)
    dummy_input = torch.rand(2, *input_s) # Giá trị pixel ngẫu nhiên từ 0-255 (trước khi chuẩn hóa trong forward)
    dummy_input = (dummy_input * 255).byte() # Chuyển sang kiểu byte để mô phỏng ảnh gốc

    print("\nDummy input shape:", dummy_input.shape)

    # Thực hiện forward pass
    q_values = model(dummy_input)
    print("Output Q-values shape:", q_values.shape) # Sẽ là (2, 5)
    print("Output Q-values:\n", q_values)