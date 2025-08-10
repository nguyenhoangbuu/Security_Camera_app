# Security Camera Model
Mô hình camera giám sát phát hiện và phân loại đối tượng.
## Model
Xây dựng mô hình phát hiện và nhận diện người từ hình ảnh thu được của camera. <br>
Nếu phát hiện người lạ xâm nhập tín hiệu ngay lập tức được phát. <br>
Sử dụng các thư viện như: tensorflow, opencv-python, ultralytics, joblib, mtcnn, urllib, tim, asyncio, nest_asyncio. <br>
![screenshot](https://github.com/nguyenhoangbuu/Security_Camera_app/blob/main/image_app.png)
## Sử dụng
Sở hữu và lắp đặt camera ESP32 tại vị trí thích hợp. <br>
Truyền hình ảnh thu được từ ESP32 về máy tính thông qua wifi với Arduino IDE. <br>
Chạy mô hình phân loại đối tượng:<br>
`python model.py`
