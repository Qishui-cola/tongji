from ultralytics import YOLO
import cv2
import numpy as np
import mss
import math
import cvzone
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk



def open_window1():
    # 开始检测
    def start_detection1():
        classNames = ['fire', 'smoke']
        myColor = (0, 255, 255)
        confThreshold = 0.4
        num = 0

        # 获取视频路径
        video_path = entry_path.get()
        
        # 创建视频捕捉对象
        cap = cv2.VideoCapture(video_path)
        
        # 创建YOLO模型
        module = YOLO(r"E:\Document\python大作业\fire_detector.pt")
        
        # 获取视频的宽度和高度
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 创建一个全屏窗口
        cv2.namedWindow("Fire Detection", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Fire Detection", cv2.WND_PROP_FULLSCREEN-50, cv2.WINDOW_FULLSCREEN-30)

        # 视频检测逻辑
            
        while True:
            success, img = cap.read()
            if not success:
                break

            results = module(img, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # 获取置信度
                    conf = math.ceil(box.conf[0] * 100) / 100
                

                    # 获取类别
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    # print(currentClass)
                    print(f"{currentClass} {conf}")

                    # 添加置信度判断，避免出现太多重复边框以及错误判断
                    if conf >= confThreshold:
                        # 在图像中显示
                        # 火焰与烟雾显示效果不同
                        if classNames[cls] == "fire":
                            myColor = (0, 0, 255)
                            cvzone.putTextRect(img, f"{currentClass} {conf}",
                                            (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                            (0, 255, 255), colorR=(0, 0, 0))
                            cvzone.cornerRect(img, (x1, y1, w, h), l=3, colorR=myColor, t=2, rt=2)
                        else:
                            myColor = (0, 255, 255)
                            cvzone.putTextRect(img, f"{currentClass} {conf}",
                                            (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                            (0, 255, 255), colorR=(0, 0, 0))
                            cvzone.cornerRect(img, (x1, y1, w, h), l=30, colorR=myColor, t=5)

            # 在全屏窗口中显示视频帧
            cv2.imshow("Fire Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        
    
    # 浏览功能
    def browse_file():
        filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4"), ("Image files", "*.jpg;*.jpeg;*.png")])
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filename)

    # 创建新窗口
    window1 = tk.Toplevel(root)
    window1.title("fire&smoke视频/图片检测")

    # 创建标签
    label_path = tk.Label(window1, text="请输入视频地址：")
    label_path.pack()

    # 创建文本框
    entry_path = tk.Entry(window1, width=50)
    entry_path.pack()

    # 创建浏览按钮
    button_browse = tk.Button(window1, text="浏览", command=browse_file)
    button_browse.pack()

    # 创建开始检测按钮
    button_start1 = tk.Button(window1, text="开始检测", command=start_detection1)
    button_start1.pack()

    

# 定义初始框选区域的起点和终点
top_left_pt = None
bottom_right_pt = None
cropping = False

def open_window2():
    # 初始化目标检测模型和参数
    module = YOLO(r"E:\Document\python大作业\fire_detector.pt")
    classNames = ['fire', 'smoke']
    confThreshold = 0.4

    # 准备开始框选屏幕区域
    print("请框选你希望检测的屏幕区域，并按Enter键确认...")
    sct = mss.mss()
    sct_img = sct.grab(sct.monitors[0])  # 获取第一个显示器的截图
    frame = np.array(sct_img)


    # 框选区域的回调函数
    def draw_rectangle(event, x, y, flags, param):
        global top_left_pt, bottom_right_pt, cropping
        # 如果是鼠标左键按下，则记录起点坐标并开启裁剪模式
        if event == cv2.EVENT_LBUTTONDOWN:
            top_left_pt = (x, y)
            cropping = True

        # 如果裁剪模式开启，且鼠标移动，则记录终点坐标并绘制矩形
        elif event == cv2.EVENT_MOUSEMOVE and cropping:
            bottom_right_pt = (x, y)
            cv2.rectangle(frame, top_left_pt, bottom_right_pt, (0, 255, 0), 2)

        # 如果鼠标左键松开，则记录终点坐标并关闭裁剪模式
        elif event == cv2.EVENT_LBUTTONUP:
            bottom_right_pt = (x, y)
            cropping = False

    # 注册回调函数
    cv2.namedWindow("Select Screen Area")
    cv2.setMouseCallback("Select Screen Area", draw_rectangle)

    # 循环直到用户按下Enter键确认
    while True:
        cv2.imshow("Select Screen Area", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter键的ASCII码值为13
            break

    # 确定框选区域的左上角和右下角坐标
    x1 = min(top_left_pt[0], bottom_right_pt[0])
    y1 = min(top_left_pt[1], bottom_right_pt[1])
    x2 = max(top_left_pt[0], bottom_right_pt[0])
    y2 = max(top_left_pt[1], bottom_right_pt[1])

    # 根据框选区域确定mon字典的参数
    mon = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}

    # 销毁窗口
    cv2.destroyAllWindows()

    # 获取视频的宽度和高度
    frame_width = mon["width"]
    frame_height = mon["height"]

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

    # 开始视频捕获和处理
    while True:
        # 捕获屏幕帧
        img = sct.grab(mon)
        img = np.array(img)
        #因为YOLO无法直接处理np的截图，会导致通道报错，所以先转化为cv的RGB颜色通道格式,再转化回到BGR格式，即可以调整为YOLO可以检测的格式，也不会因为红蓝通道调换而出现图片色差不对的情况
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # 进行目标检测
        results = module(img2, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # 获取置信度
                conf = math.ceil(box.conf[0] * 100) / 100
            

                # 获取类别
                cls = int(box.cls[0])
                currentClass = classNames[cls]
                # print(currentClass)
                print(f"{currentClass} {conf}")

                # 添加置信度判断，避免出现太多重复边框以及错误判断
                if conf >= confThreshold:
                    # 在图像中显示
                    # 火焰与烟雾显示效果不同
                    if classNames[cls] == "fire":
                        myColor = (0, 0, 255)
                        cvzone.putTextRect(img, f"{currentClass} {conf}",
                                        (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                        (0, 255, 255), colorR=(0, 0, 0))
                        cvzone.cornerRect(img, (x1, y1, w, h), l=3, colorR=myColor, t=2, rt=2)
                    else:
                        myColor = (0, 255, 255)
                        cvzone.putTextRect(img, f"{currentClass} {conf}",
                                        (max(0, x1 + 10), max(35, y1 - 10)), 1, 1,
                                        (0, 255, 255), colorR=(0, 0, 0))
                        cvzone.cornerRect(img, (x1, y1, w, h), l=30, colorR=myColor, t=5)

        # 在视频中写入帧，不需要
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #out.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 在窗口中显示视频帧
        cv2.imshow("Fire Detection", img)
        
        # 检测用户是否按下了 'q' 键，如果按下则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    out.release()
    cv2.destroyAllWindows()






# 创建主窗口
root = tk.Tk()
root.title("模式选择")

# 创建标签
label_path = tk.Label(root, text="请选择模式")
label_path.pack()

# 创建按钮1，点击后打开窗口1
button1 = tk.Button(root, text="选择视频/图片路径", command=open_window1)
button1.pack()

# 创建按钮2，点击后打开窗口2
button2 = tk.Button(root, text="自主框选屏幕检测范围", command=open_window2)
button2.pack()


# 运行主循环
root.mainloop()