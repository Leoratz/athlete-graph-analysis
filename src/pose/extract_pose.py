import cv2

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la vidéo")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Vidéo chargée :")
    print(f"  FPS : {fps}")
    print(f"  Résolution : {width}x{height}")
    print(f"  Nombre de frames : {total_frames}")
    
    cap.release()

def show_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (640, 360))
        cv2.imshow('Video', frame_resized)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_video("data/test.mp4")
    show_video("data/test.mp4")