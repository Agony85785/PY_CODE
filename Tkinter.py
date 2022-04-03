# 姓名 ：李星宇
# 开发时间 : 2022/4/1 23:29
import tkinter as tk
from tkinter import *
from tkinter import messagebox

# GUI：图形化界面
window = tk.Tk()  # 创建窗口  （所有object前面的一个字符是要大写的）
window.title("加法程序")  # 设置窗口名字
window.geometry('400x100')  # 设置窗口大小，宽乘以高，中间是乘号
# l = tk.Label(window, text = 'OMG!this is TK!', bg= 'green', font=('Arial, 12'),width=15,height=2 )
# 设置标签，text为文字,bg为背景,font为字体,字体大小为12，width=15,height=2为字符的宽高
L1 = tk.Label(window, text="数1")
L1.place(x=20, y=20)  # 将标签放在（20,20）处
E1 = Entry(window)  # 输入控件；用于显示简单的文本内容。
E1.place(x=50, y=20)  # 将输入框放在（50,20）处
L2 = tk.Label(window, text="数2")
L2.place(x=220, y=20)
E2 = Entry(window)
E2.place(x=250, y=20)


def helloCallBack():
    mystr = eval(E1.get()) + eval(E2.get())
    msg = messagebox.showinfo("结果", str(mystr))  # 设置消息窗口，第一个参数为内容标题，第二个参数为内容


B = Button(window, text="相加结果：", command=helloCallBack)  # 按钮  command为点击之后执行的函数
B.place(x=100, y=50)

# b = tk.Button(window, text = 'hit me',width=15, height=2,command=hit_me)  # command为点击之后执行的函数
window.mainloop()
