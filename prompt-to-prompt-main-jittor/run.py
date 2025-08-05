import sys
import io
import base64
import datetime
import importlib.util  # 用于导入含横杠的文件名
from PIL import Image
from contextlib import redirect_stdout
import matplotlib

matplotlib.use('Agg')  # 强制matplotlib不显示图像，只保存
import matplotlib.pyplot as plt

# 全局变量：存储捕获的图像和文本
captured_images = []
captured_text = []


# 1. 替换matplotlib的show函数，捕获图像
def capture_plt_show():
    original_show = plt.show

    def new_show(*args, **kwargs):
        # 将图像保存到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        captured_images.append(img)
        # 清空当前图，避免重叠
        plt.close()

    plt.show = new_show  # 替换原函数


# 2. 捕获print等文本输出
class TextCapture:
    def __enter__(self):
        self.buffer = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        captured_text.append(self.buffer.getvalue())


# 3. 运行原程序（处理含横杠的文件名导入）
def run_original_script():
    # 导入含横杠的文件：prompt-to-prompt_stable.py
    spec = importlib.util.spec_from_file_location(
        "prompt_module",  # 给模块起一个临时名称
        "prompt-to-prompt_stable.py"  # 实际文件名
    )
    prompt_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompt_module)  # 执行原脚本


# 4. 生成HTML文件
def generate_html():
    # 转换图像为base64
    img_tags = []
    for i, img in enumerate(captured_images):
        buf = io.BytesIO()
        img.save(buf, format='png')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        img_tags.append(f"""
        <div style="margin: 20px 0;">
            <h3>生成图像 {i + 1}</h3>
            <img src="data:image/png;base64,{img_b64}" style="max-width: 800px; border: 1px solid #ddd; padding: 5px;">
        </div>
        """)

    # 组合文本输出
    text_output = "".join(
        [f"<pre style='background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 10px 0;'>{t}</pre>" for t in
         captured_text])

    # 完整HTML内容
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>程序输出 - {datetime.datetime.now().strftime('%Y%m%d')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            h1 {{ color: #333; border-bottom: 2px solid #666; padding-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>程序运行结果</h1>
        <p>生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <div id="text-output">{text_output}</div>
        <div id="images">{''.join(img_tags)}</div>
    </body>
    </html>
    """

    # 保存HTML
    filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"结果已保存到: {filename}")


# 主流程
if __name__ == "__main__":
    capture_plt_show()  # 替换图像显示函数
    with TextCapture():  # 捕获文本输出
        run_original_script()  # 运行原程序
    generate_html()  # 生成HTML
