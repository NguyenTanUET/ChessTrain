[![Watch the video](https://img.youtube.com/vi/IVZYHFL5thE/maxresdefault.jpg)](https://youtu.be/IVZYHFL5thE)

### [Video Demo](https://youtu.be/IVZYHFL5thE)

<div align="center">

# bullet

</div>

Một thư viện học máy chuyên dụng, thường được sử dụng để huấn luyện các mạng NNUE cho một số engine cờ vua mạnh nhất trên thế giới.
Core framework của hệ thống (folder crates) được tham khảo từ repository:

```toml
bullet = { git = "https://github.com/jw1912/bullet", package = "bullet_lib" }
```
### Giải thích NNUE và ứng dụng của NNUE trong chess training
NNUE (Efficiently Updatable Neural Network) là một mạng thần kinh nhỏ gọn nhưng vẫn đủ mạnh để đánh giá thế cờ vua (hoặc shogi) trên CPU với tốc độ rất cao, phù hợp nhúng trực tiếp vào engine.
1. Biểu diễn input

- Mỗi vị thế bàn cờ được mã hoá thành một vector 768 chiều nhị phân (6 loại quân × 64 ô × 2 màu).

- Không dùng convolution(CNN) hay attention(Transformer)—chỉ đơn giản là “bit is set” hay “bit is clear”, đảm bảo được hiệu năng cao trên CPU.

2. Accumulator & cập nhật hiệu quả

- Thay vì tính toán toàn bộ mạng sau mỗi nước đi, NNUE duy trì một accumulator (lớp ẩn) có thể cập nhật nhanh:

    - Khi một quân di chuyển, chỉ một số rất nhỏ bit input thay đổi (thường 2–4 neurons).

    - Chúng ta “trừ” đi đóng góp của các neurons tắt và “cộng” đóng góp của các neurons bật mới, tránh phải tính lại toàn bộ ma trận.

3. Hai “perspectives” trắng/đen

- NNUE giữ hai accumulator song song:

    - White perspective: tính như bình thường.

    - Black perspective: trước khi cập nhật, “lật” (mirror) bàn cờ và đảo màu quân để luôn tính dưới góc nhìn như đối trắng.

- Khi đánh giá, ghép hai accumulator lại, đảm bảo mạng nắm đủ thông tin về bên đang đi.

4. Kích hoạt & quantization

- Sử dụng hàm kích hoạt SCReLU (Squared Clipped ReLU) để giới hạn và bình phương giá trị hidden, giúp ổn định học và inference.

- Toàn bộ weights và biases được quantize (thường sang int16) qua hai hệ số QA, QB, cho phép tính toán chỉ bằng phép cộng và nhân số nguyên, rất nhanh trên CPU.

5. Các cải tiến mở rộng

- Output buckets: chia output thành nhiều nhóm (buckets) theo số quân còn lại, mỗi nhóm có weight riêng để mạng đánh giá chuyên biệt cho từng giai đoạn material.

- Horizontal mirroring: luôn đặt vua bên trái trước khi cập nhật, vừa là data-augmentation, vừa giúp mạng học ổn định hơn mối quan hệ vị trí vua–các quân còn lại.

### Crates

- **bullet_core**
  - Tạo và quản lý đồ thị neural network
  - Thực hiện forward và backward propagation
  - Tối ưu hóa trọng số mạng
  - Backend CPU đơn luồng để verify
- **bullet_cuda_backend**
  -  CUDA implementation cho NVIDIA GPUs
- **bullet_hip_backend**
  - HIP implementation cho AMD GPUs
- **bullet_lib**
  - High-level API cho training
  - Quản lý các tham số training
  - Data loading và processing
  - Định nghĩa kiến trúc mạng NNUE
- **bullet-utils**
    - Là các tiện ích hộ trợ:
      - Data handling utilities
      - File I/O operations
      - Training data format conversions
      - Debugging tools

### Các Model được dùng để huấn luyện:
Các model được nhóm tái sử dụng và thay đổi cho phù hợp từ repository gốc, được chia làm 4 mô hình với các điểm chung sau:
- Đều dùng bullet_lib để xây dựng và chạy trainer
- Sử dụng kiến trúc NNUE cơ bản, với các đặc tính như:
  - Input dimension cố định 768 (tương ứng 12 loại quân × 64 ô). 
  - Hai accumulator song song (perspective ×2: trắng/đen). 
  - Một lớp ẩn (feature transformer) kích thước HIDDEN_SIZE (512 trong các bản thử nghiệm, 1024 ở bản cuối). 
  - Output heads chia theo 8 buckets dựa trên material balance (MaterialCount<OUTPUT_BUCKETS>).
- Loss function dùng SigmoidMSE (tối thiểu hoá mean-squared error trên sigmoid outputs). 
- Dữ liệu được lấy từ data/training_data.binpack (~120Gb)

Ngoài những điểm giống nhau trên, các model được thay đổi từ ver1 -> ver4, mỗi bản được cải thiện hơn so với bản trước, được ghi trong bảng so sánh sau:

| Phiên bản           | Hidden size | Output buckets | Input type             | Mirroring | LR-step | Threads | Save rate | Mục đích chính                                                                        |
|---------------------| ----------- | -------------- | ---------------------- | --------- | ------- | ------- | --------- | ------------------------------------------------------------------------------------- |
| **simple\_ver1.rs** | 512         | —              | `Chess768`             | No        | 8       | 6       | 20        | **Baseline**: mạng nhỏ (512), scalar output đơn, để đánh giá hiệu năng khởi đầu .     |
| **simple\_ver2.rs** | 512         | 8              | `Chess768`             | No        | 50      | 6       | 10        | **Thêm Material-buckets**: phân 8 đầu ra theo material balance, giữ input gốc .       |
| **simple\_ver3.rs** | 512         | 8              | `ChessBucketsMirrored` | Yes       | 200     | 8       | 10        | **Buckets + Mirroring**: augment dữ liệu qua mirror, chuyên gia tăng generalization . |
| **simple\_ver4.rs** | 1024        | 8              | `ChessBucketsMirrored` | Yes       | 200     | 8       | 10        | **Final-tuning**: mở rộng capacity, resume từ sb 240, fine-tune trên FourthModel .    |
