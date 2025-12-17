from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import joblib
import os
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- THIẾT LẬP BIẾN TOÀN CỤC VÀ LOCK ---
# Bộ nhớ chia sẻ toàn cục để lưu trữ kết quả xử lý của các thư mục
PROCESSING_RESULTS = {}
# Lock để đảm bảo an toàn luồng khi đọc/ghi biến PROCESSING_RESULTS
RESULTS_LOCK = threading.Lock()

# THIẾT LẬP SEMAPHORE ĐỂ GIỚI HẠN XỬ LÝ ĐỒNG THỜI
# Chỉ cho phép tối đa 2 luồng chạy YOLO/RF cùng lúc
MAX_CONCURRENT_PROCESSES = 1
PROCESS_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_PROCESSES) 


# --- THIẾT LẬP ĐƯỜNG DẪN VÀ LOAD MODELS ---
app = Flask(__name__)
# Đổi tên thư mục gốc chứa các folder con mà hệ thống sẽ giám sát
MONITORED_ROOT_FOLDER = "monitored_folders"
os.makedirs(MONITORED_ROOT_FOLDER, exist_ok=True)

# SỬA LỖI ĐƯỜNG DẪN: Sử dụng os.path.join và đường dẫn tương đối an toàn
# Giả định 'yolov8x.pt' nằm cùng thư mục với 'app.py'
YOLO_MODEL_PATH = os.path.join(os.getcwd(), "yolov8x.pt")
# Giả định scaler và rf_model nằm trong thư mục models/ (ngang hàng với app.py)
SCALER_PATH = os.path.join("models", "scaler_rf.pkl") 
MODEL_PATH = os.path.join("models", "rf_model.pkl") 

# Load YOLO, scaler, RF model
try:
    print(f"Đang cố gắng tải mô hình YOLO từ: {YOLO_MODEL_PATH}")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    rf_model = joblib.load(MODEL_PATH)
    print("Tải model thành công.")
except FileNotFoundError as e:
    print(f"\n--- LỖI KHỞI TẠO (FILE NOT FOUND) ---")
    print(f"Không tìm thấy file model. Vui lòng kiểm tra các đường dẫn:")
    print(f"YOLO: {YOLO_MODEL_PATH}, SCALER: {SCALER_PATH}, MODEL: {MODEL_PATH}")
    exit()
except RuntimeError as e:
    # Bắt lỗi file hỏng do PyTorchStreamReader
    print(f"\n--- LỖI KHỞI TẠO (RUNTIME ERROR) ---")
    print(f"Chi tiết lỗi: {e}")
    print("Vui lòng tải lại file yolov8x.pt, vì nó có vẻ bị hỏng hoặc tải về không đầy đủ.")
    exit()
except Exception as e:
    # Bắt các lỗi khác, bao gồm cả lỗi 'Conv' object has no attribute 'bn'
    print(f"\n--- LỖI KHỞI TẠO CHUNG ---")
    print(f"Lỗi khi tải model (ví dụ: lỗi phiên bản thư viện): {e}")
    print("Nếu gặp lỗi 'Conv', vui lòng cập nhật hoặc hạ cấp thư viện ultralytics/pytorch.")
    exit()


VEHICLE_CLASSES = {'motorcycle': 'Motorcycles', 'car': 'Cars', 'truck': 'Trucks'}

# --- HÀM DỰ ĐOÁN (Giữ nguyên logic gốc) ---
def predict_congestion_status(yolo_counts):
    mc = yolo_counts.get('Motorcycles', 0)
    cars = yolo_counts.get('Cars', 0)
    trucks = yolo_counts.get('Trucks', 0)
    total = mc + cars + trucks
    
    if total == 0:
        truck_car_percentage = 0
        return False, 0.00

    truck_car_percentage = ((cars + trucks) / total) * 100

    # sử dụng 5 features như lúc train
    X_new = np.array([[total, mc, cars, trucks, truck_car_percentage]])
    X_scaled = scaler.transform(X_new)
    prob_congested = rf_model.predict_proba(X_scaled)[:, 1][0]
    is_congested = prob_congested >= 0.3
    return is_congested, prob_congested


# --- LOGIC XỬ LÝ ẢNH KHI THƯ MỤC MỚI ĐƯỢC TẠO (Đã sửa Polling và Semaphore) ---
def process_new_folder(folder_path):
    """Quét và xử lý tất cả ảnh trong thư mục mới được tạo, sử dụng polling để chờ file ổn định."""
    folder_name = os.path.basename(folder_path)
    print(f"\n--- Bắt đầu xử lý thư mục mới: {folder_name} ---")
    folder_results = []
    
    # --- PHẦN 1: Chờ file xuất hiện và ổn định (Polling) ---
    MAX_WAIT_TIME = 60 
    POLL_INTERVAL = 1 
    start_time = time.time()

    image_files = []
    while time.time() - start_time < MAX_WAIT_TIME:
        try:
            current_files = os.listdir(folder_path)
            image_files = [f for f in current_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if image_files:
                all_files_ready = True
                for filename in image_files:
                    filepath = os.path.join(folder_path, filename)
                    try:
                        # Kiểm tra I/O Lock/Đang ghi
                        with open(filepath, 'rb') as f:
                            f.read(1) 
                        # Kiểm tra tính toàn vẹn ảnh bằng CV2
                        if cv2.imread(filepath) is None:
                            all_files_ready = False
                            print(f"File {filename} chưa đọc được hoặc chưa hoàn tất (CV2). Đợi...")
                            break
                    except IOError:
                        all_files_ready = False
                        print(f"File {filename} đang được ghi. Đợi...")
                        break
                
                if all_files_ready:
                    print(f"Đã phát hiện và xác nhận {len(image_files)} file ảnh. Bắt đầu xử lý.")
                    break 
            
            print(f"Đang chờ file ảnh xuất hiện hoặc ổn định trong {folder_name}...")
            time.sleep(POLL_INTERVAL)
            
        except FileNotFoundError:
             print(f"Lỗi: Thư mục {folder_name} không tồn tại trong quá trình chờ.")
             return 
    
    if not image_files and time.time() - start_time >= MAX_WAIT_TIME:
        print(f"CẢNH BÁO: Hết thời gian chờ. Không tìm thấy file ảnh hợp lệ nào trong thư mục {folder_name}.")
        return

    # --- PHẦN 2: BẮT ĐẦU XỬ LÝ ẢNH (Bảo vệ bằng Semaphore) ---
    # Luồng sẽ đợi ở đây nếu đã có 2 luồng khác đang xử lý (nhờ PROCESS_SEMAPHORE)
    with PROCESS_SEMAPHORE: 
        print(f"Luồng xử lý {folder_name} đã được cấp quyền và BẮT ĐẦU xử lý mô hình.")
        for filename in image_files:
            filepath = os.path.join(folder_path, filename)
            
            frame = cv2.imread(filepath)
            if frame is None:
                folder_results.append({
                    "filename": filename, 
                    "status": "Lỗi đọc ảnh", 
                    "probability": 0.00,
                    "counts": {'Total_Vehicles': 0, 'Motorcycles': 0, 'Cars': 0, 'Trucks': 0},
                    "error": "Không đọc được ảnh CV2"
                })
                continue

            try:
                # 1. Phát hiện đối tượng bằng YOLO
                detections = yolo_model(frame, verbose=False)[0].boxes
                frame_counts = {v: 0 for v in VEHICLE_CLASSES.values()}

                for box in detections:
                    class_id = int(box.cls[0])
                    yolo_class_name = yolo_model.names[class_id]
                    if yolo_class_name in VEHICLE_CLASSES:
                        rf_class_name = VEHICLE_CLASSES[yolo_class_name]
                        frame_counts[rf_class_name] += 1

                # 2. Dự đoán kẹt xe bằng RF Model
                is_congested, prob = predict_congestion_status(frame_counts)
                
                # BỔ SUNG: Tính tổng số xe và thêm vào counts để JS đọc
                total_vehicles = sum(frame_counts.values()) 
                frame_counts['Total_Vehicles'] = total_vehicles 
                
                # 3. Lưu kết quả
                folder_results.append({
                    "filename": filename,
                    "status": "Kẹt xe" if is_congested else "Bình thường",
                    "probability": round(prob, 2),
                    "counts": frame_counts
                })
            except Exception as e:
                # CẤU TRÚC LỖI AN TOÀN
                print(f"LỖI XỬ LÝ ẢNH {filename} trong thư mục {folder_name}: {e}")
                
                folder_results.append({
                    "filename": filename,
                    "status": "Lỗi xử lý",         
                    "probability": 0.00,        
                    "counts": {'Total_Vehicles': 0, 'Motorcycles': 0, 'Cars': 0, 'Trucks': 0}, 
                    "error": f"Lỗi: {e}" 
                })

    # --- LƯU KẾT QUẢ VÀO BỘ NHỚ CHIA SẺ VỚI LOCK ---
    with RESULTS_LOCK:
        PROCESSING_RESULTS[folder_name] = folder_results

    # In/Lưu kết quả
    print(f"--- Kết quả xử lý thư mục {folder_name} đã được lưu ---")
    for res in folder_results:
        status_display = res.get('status', 'N/A')
        if 'error' in res:
             status_display = f"{status_display} ({res['error'].split(':')[0]})"
        print(f"  - {res['filename']}: {status_display} (P: {res.get('probability', 'N/A')})")
    print(f"--- Xử lý {folder_name} hoàn tất. Semaphore đã được giải phóng. ---")


# --- LOGIC GIÁM SÁT THƯ MỤC (WATCHDOG) ---
class FolderMonitorHandler(FileSystemEventHandler):
    """Xử lý các sự kiện hệ thống file, kích hoạt xử lý khi thư mục mới được tạo."""
    def on_created(self, event):
        # Chỉ xử lý khi một thư mục (directory) được tạo
        if event.is_directory:
            # Chạy xử lý trong một luồng riêng
            threading.Thread(target=process_new_folder, args=(event.src_path,)).start()

def start_folder_monitoring():
    """Thiết lập và khởi động Observer để giám sát MONITRED_ROOT_FOLDER."""
    event_handler = FolderMonitorHandler()
    observer = Observer()
    observer.schedule(event_handler, MONITORED_ROOT_FOLDER, recursive=False)
    observer.start()
    print(f"\n--- HỆ THỐNG GIÁM SÁT ĐANG CHẠY ---")
    print(f"Đang theo dõi thư mục: {MONITORED_ROOT_FOLDER}")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# --- PHẦN FLASK (Giao diện web và API Kết quả) ---

@app.route("/")
def index():
    # Trang web sẽ dùng JavaScript để gọi API /results
    # Nếu bạn chưa có file index.html trong thư mục 'templates', bạn cần tạo nó.
    return render_template("index.html")

@app.route("/results", methods=["GET"])
def get_results():
    """Endpoint API để giao diện web truy vấn kết quả xử lý một cách an toàn luồng."""
    with RESULTS_LOCK:
        # Trả về bản sao của dữ liệu. Sử dụng .copy() là an toàn nhất.
        return jsonify(PROCESSING_RESULTS.copy())

@app.route("/upload", methods=["POST"])
def upload_files():
    # Endpoint này chỉ thông báo cho người dùng
    return jsonify({"message": f"Hệ thống đang hoạt động. Vui lòng tạo thư mục con chứa ảnh trong thư mục '{MONITORED_ROOT_FOLDER}' để kích hoạt xử lý tự động."}), 200

# --- KHỞI CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    # Khởi chạy giám sát thư mục trong một luồng nền
    monitoring_thread = threading.Thread(target=start_folder_monitoring, daemon=True)
    monitoring_thread.start()
    
    # Khởi chạy ứng dụng Flask
    # use_reloader=False là cần thiết để tránh việc thread giám sát bị chạy 2 lần.
    app.run(debug=True, use_reloader=False)