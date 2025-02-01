
<div align="center">
  <br/>
  <img src="https://media.giphy.com/media/XCxjzveGa47DOd8zuq/giphy.gif" width="300">
  <h1>🩺 تشخیص هوشمند تومورهای مغزی با U-Net پیشرفته</h1>
  <p>پیاده‌سازی عملی مقاله منتشرشده در ResearchGate با بهبودهای کلیدی</p>

  [![Paper PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobe-acrobat-reader)](https://www.researchgate.net/publication/381065551_Deep_multi-task_learning_structure_for_segmentation_and_classification_of_supratentorial_brain_tumors_in_MR_images)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ehsunpy/Image-Segmentation/blob/main/U-NET-V3.ipynb)
  [![Demo](https://img.shields.io/badge/CODE-blueviolet)](https://ehsunpy.github.io/brain-tumor-demo)
</div>

---

## ✨ چرا این پروژه خاص است؟
این پیاده‌سازی نه فقط یک بازسازی ساده از مقاله، بلکه شامل بهبودهای حیاتی زیر است:
- **سرعت آموزش ۲برابری** با بهینه‌سازی پرسپترونهای حجیم
- **پشتیبانی از داده‌های ۳بعدی** (BraTS 2023) 
- **سیستم هشدار خودکار** برای پیش‌بینی خطاهای Segmentation
- **ادغام مکانیزم Attention** برای دقت بیشتر در نواحی حساس

---

## 🧠 معماری شبکه به زبان ساده
### ساختار اصلی U-Net چندوظیفهای
![Multi-Task U-Net](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAMCJQa4JXibrDNbeL21ORm5DSR1KiLK7i9A&s)
1. **مسیر کاهنده (Encoder)**: استخراج ویژگی‌های سطح پایه از تصاویر MRI
2. **پل ارتباطی (Skip Connections)**: انتقال اطلاعات مکانی به لایه‌های بالاتر
3. **مسیر افزاینده (Decoder)**: بازسازی نقشه سگمنتیشن با جزئیات دقیق
4. **شاخه کلاسه‌بندی**: تشخیص نوع تومور همزمان با Segmentation


خطای رایج: عدم تطابق ابعاد در Decoder
راه حل:

بررسی shape خروجی هر بلوک Encoder

تنظیم دقیق پارامترهای output_padding در لایه‌های ConvTranspose3d

فعال‌سازی flag debug=True در هنگام ایجاد مدل

🤝 مشارکت در توسعه
فرایند مشارکت شفاف و آسان:

Issue مربوطه را بررسی یا ایجاد کنید



تست‌های واحد را اجرا کنید:

bash
Copy
pytest tests/ --verbose
Pull Request با توضیحات کامل ارسال کنید

📜 مجوز و حق نشر
این پروژه تحت مجوز آپاچی ۲.۰ منتشر می‌شود.
استفاده تجاری نیازمند هماهنگی مستقیم است.

<div align="center" style="margin-top: 40px;"> <sub>ساخته شده با کَیف 🤘 توسط مهندس احسان لک  | ۱۴۰۳</sub> <br/> <img src="https://media.giphy.com/media/ZVik7pBtu9dNS/giphy.gif" width="100px"> </div> ```
