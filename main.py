import hubconf
import cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def car_detection(model, video_path):
    # To capture video from existing video.   
    cap = cv2.VideoCapture(video_path)  
  
    while True:  
        # Read the frame  
        _, img = cap.read()
    
        # Detection
        results = model(img)
        print(results.xyxyn)

        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
 
        n = len(labels)
        x_shape, y_shape = img.shape[1], img.shape[0]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 1)
            label = f"{int(row[4])*100}"
            name = {label:'person'}
            cv2.putText(img, name[label], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            

        cv2.putText(img, f"Total People: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Display  
        cv2.imshow('Video', img)
    
        k = cv2.waitKey(10)                           
        
        # check if key is q then exit
        if k == ord("q"):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(results.pandas().xyxy)
    print(results.pandas().xyxy[0])


if __name__ == "__main__":

    model = hubconf.custom(path='best_obj_track.pt')
    car_detection(model=model, video_path='/Users/azureennaja/Desktop/Personal Projectto/footfall_count/yolo/yolov5/video2.mp4')



