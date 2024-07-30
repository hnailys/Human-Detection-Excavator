import numpy as np
import cv2

# Kelas deteksi objek
classes = ["1"]

# Daftar video yang akan diproses
video_files = [
    "ReferenceVideos/video1.mp4",
    "ReferenceVideos/video2.mp4",
    "ReferenceVideos/video3.mp4"
]

# Menggunakan model YOLO dari file ONNX
net = cv2.dnn.readNetFromONNX("best.onnx")

# Fungsi untuk menggambar kotak dan jarak
def draw_boxes_and_distances(frame, boxes, confidences, label, distances):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    height, width_panel, _ = frame.shape

    # Membuat panel kanan untuk info jarak
    width_panel = 200
    right_panel = np.zeros((height, width_panel, 3), dtype=np.uint8)

    humans_detected = len(boxes)
    close_detections = sum(d < 3 for d in distances)
    
    # Menambahkan jumlah manusia terdeteksi dan deteksi dalam jarak kurang dari tiga meter
    summary_text = f"Humans detected: {humans_detected}"
    close_summary_text = f"Close detections (<3m): {close_detections}"
    cv2.putText(frame, summary_text, (10, 20), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(frame, close_summary_text, (10, 50), font, font_scale, (255, 255, 255), thickness)

    for i, box in enumerate(boxes):
        x1, y1, w, h = box
        conf = confidences[i]
        label = f'{i+1}: {conf:.2f}'  # Memberi nomor pada objek terdeteksi
        distance = distances[i]
        distance_text = f'{label} = {distance:.2f} m'

        # Memberi warna kotak berdasarkan jarak
        box_color = (0, 255, 0) if distance >= 3 else (0, 0, 255)
        # Memberi warna teks jarak berdasarkan jarak
        text_color = (0, 255, 0) if distance >= 3 else (0, 0, 255)

        # Menggambar kotak
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), box_color, thickness)
       
        # Menambahkan label nomor pada kotak
        cv2.putText(frame, label, (x1, y1 - 10), font, font_scale, box_color, thickness)
        
        # Menambahkan info jarak pada panel kanan
        cv2.putText(frame, distance_text, (10, 80 + i * 20), font, font_scale, text_color, thickness)

    # Menggabungkan frame asli dengan panel kanan
    combined_frame = np.hstack((frame, right_panel))

    return combined_frame, close_detections

total_humans_detected = 0  # Inisialisasi penghitung deteksi manusia
close_detections = 0  # Inisialisasi penhitung deteksi manusia di jarak kurang dari 3 meter

for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    frame_count = 0  # Inisialisasi penghitung frame untuk setiap video

    while frame_count < 600:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        # Mendeteksi objek
        height, width, channels = frame.shape 
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()[0]

        classes_ids = []
        confidences = []
        boxes = []
        distances = []
        labels = []
        rows = detections.shape[0]

        frame_width, frame_height = frame.shape[1], frame.shape[0]
        x_scale = frame_width / 640
        y_scale = frame_height / 640

        # Memproses hasil deteksi
        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.2:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2) * x_scale)
                    y1 = int((cy - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = [x1, y1, width, height]
                    boxes.append(box)
                    
                    # Menghitung jarak 
                    distance = ((2 * 3.14 * 180) / (w + h * 360) * 1000) - 9
                    distances.append(distance)
                    labels.append(f'{ind}: {confidence:.2f}')

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)

        final_boxes = []
        final_confidences = []
        final_labels = []
        final_distances = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_labels.append(labels[i])
                final_distances.append(distances[i])

        # Menggambar kotak dan jarak
        output_frame, close_detections_frame = draw_boxes_and_distances(frame, final_boxes, final_confidences, final_labels, final_distances)
        close_detections += close_detections_frame

        # Menghitung total manusia terdeteksi
        total_humans_detected += len(final_boxes)
        
        cv2.imshow("Frame", output_frame)
        key = cv2.waitKey(1) 
        if key == 27:  # Tekan ESC untuk keluar
            break

    cap.release()

cv2.destroyAllWindows()

print(f"Total humans detected in 1800 frames (600 per video): {total_humans_detected}")
print(f"Total close detections (<3m) in 1800 frames: {close_detections}")
