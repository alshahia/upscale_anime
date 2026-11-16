[English](README.md) | العربية

<div dir="rtl" align="right">

# واجهة مُكبِّر الأنمي & أدوات الاستمتاع الفائق للأنمي (PyTorch)

[![CI](https://github.com/alshahia/upscale_anime/actions/workflows/ci.yml/badge.svg?branch=release%2Fgui-preview)](https://github.com/alshahia/upscale_anime/actions/workflows/ci.yml?query=branch%3Arelease%2Fgui-preview)
[![الإصدار: v0.2.0-preview](https://img.shields.io/badge/release-v0.2.0--preview-orange)](https://github.com/alshahia/upscale_anime/releases/tag/v0.2.0-preview)
[![الرخصة: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB)](pyproject.toml)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-EE4C2C)](requirements.txt)

تكبير صور الأنمي باستخدام PyTorch، يأتي في شكلين: **واجهة رسومية مستقلة لنظام Windows** لتكبير الصور ومقاطع الفيديو، و**حزمة بحثية/تدريبية** (`anime_sr`) للاستمتاع الفائق (super-resolution) للأنمي باستخدام SPAN وMamba-PAN وتقطير المعرفة التدريجي للتجميع (progressive ensemble distillation) إلى طلاب خفيفين من نوع RFDN/TinySRVGG. يشترك الشكلان في نفس شيفرة النماذج ومنظومة الاستدلال على المعالج (CPU) ووحدات معالجة الرسومات (CUDA)، مع معالجة دفعية بقائمة انتظار، واستدلال مقسَّم إلى بلاطات لتفادي نقص ذاكرة VRAM، وملفات إعدادات قابلة لإعادة الإنتاج.

> **الحالة:** نسخة تجريبية مسبقة (preview). الواجهة الرسومية متاحة عبر الإصدار المسبق على GitHub [`v0.2.0-preview`](https://github.com/alshahia/upscale_anime/releases/tag/v0.2.0-preview)؛ أما الحزمة البحثية فهي شيفرة ألفا (alpha) وليست منشورة على PyPI. راجع [الحالة والقيود](#الحالة-والقيود).

## المحتويات

- [شيئان في مستودع واحد](#شيئان-في-مستودع-واحد)
- [تنزيل الواجهة الرسومية وتثبيتها (Windows)](#تنزيل-الواجهة-الرسومية-وتثبيتها-windows)
- [ميزات الواجهة الرسومية](#ميزات-الواجهة-الرسومية)
- [النماذج المدعومة](#النماذج-المدعومة)
- [البحث: التدريب والتقطير](#البحث-التدريب-والتقطير)
- [قياسات الأداء والنتائج المقاسة](#قياسات-الأداء-والنتائج-المقاسة)
- [للباحثين](#للباحثين)
- [لوكلاء الذكاء الاصطناعي وأبحاث النماذج اللغوية](#لوكلاء-الذكاء-الاصطناعي-وأبحاث-النماذج-اللغوية)
- [الإعدادات](#الإعدادات)
- [التطوير](#التطوير)
- [بنية المستودع](#بنية-المستودع)
- [التوثيق](#التوثيق)
- [الحالة والقيود](#الحالة-والقيود)
- [الرخصة](#الرخصة)
- [شكر وتقدير](#شكر-وتقدير)

## شيئان في مستودع واحد

| | الواجهة الرسومية المستقلة (مُصدَرة) | الحزمة البحثية (من المصدر) |
|---|---|---|
| ما هي | تطبيق سطح مكتب بـ Tkinter لتكبير الصور ومقاطع الفيديو: قائمة انتظار، معاينة، حارس VRAM، معالجة دفعية | شيفرة `anime_sr` بـ PyTorch للتدريب والتقطير والتقييم والنشر |
| التثبيت | حزمة wheel من [إصدار v0.2.0-preview](https://github.com/alshahia/upscale_anime/releases/tag/v0.2.0-preview) | `pip install -e ".[dev]"` من نسخة المصدر |
| الإصدار | `anime-upscaler-gui 0.2.0` (إصدار مسبق) | `anime-sr 0.1.0` (ألفا، من المصدر فقط) |
| نقاط الدخول | `anime-upscaler-gui` / `anime-upscaler-gui --self-check` | واجهة `anime-sr` الطرفية، واجهة Python، أدوات `scripts/` |
| الحالة | تم اختبار الإصدار على Windows 11 | شيفرة بحثية؛ CI أخضر على Ubuntu |

## تنزيل الواجهة الرسومية وتثبيتها (Windows)

توزَّع الواجهة الرسومية كحزمة wheel واحدة على صفحة الإصدار المسبق: [`v0.2.0-preview`](https://github.com/alshahia/upscale_anime/releases/tag/v0.2.0-preview).
الملفات المرفقة: `anime_upscaler_gui-0.2.0-py3-none-any.whl` (حوالي 3.0 ميغابايت) و`SHA256SUMS.txt`.

1. نزّل حزمة wheel (وملف `SHA256SUMS.txt`) من صفحة الإصدار.
2. تحقّق من سلامة التنزيل (اختياري لكنه مستحسن):

   ```powershell
   Get-FileHash .\anime_upscaler_gui-0.2.0-py3-none-any.whl -Algorithm SHA256
   # قارن الناتج بالبصمة الموجودة في SHA256SUMS.txt
   ```

3. أنشئ بيئة افتراضية وثبّت الحزمة:

   ```powershell
   python -m venv .venv-anime-gui
   .venv-anime-gui\Scripts\python.exe -m pip install anime_upscaler_gui-0.2.0-py3-none-any.whl
   ```

4. شغّل الفحص الذاتي للبيئة، ثم شغّل التطبيق:

   ```powershell
   .venv-anime-gui\Scripts\anime-upscaler-gui.exe --self-check
   .venv-anime-gui\Scripts\anime-upscaler-gui.exe
   ```

**المتطلبات المسبقة**

- **Windows 11** (بُني الإصدار واختُبر على Windows 11).
- **Python 3.10 أو أحدث** (يُوصى بـ 3.12) لتثبيت الحزمة، أو استخدم بيئة جاهزة.
- **`ffmpeg` و`ffprobe` على `PATH`** لفك ترميز الفيديو وترميزه؛ ويُبلِّغ `--self-check` عن وجودهما.
- وحدة معالجة الرسومات اختيارية: يعمل المعالج (CPU)، وتُستخدم CUDA عند توفرها. الهدف المضبوط عليه الضبط هو وحدة رسوميات بذاكرة 8 غيغابايت.

**لا تُرفَق أوزان النماذج مع الحزمة.** تحتوي حزمة wheel على شيفرة النماذج وكتالوج الإعدادات المسبقة فقط؛ وتُنزَّل نقاط التفتيش أو تُستورَد عبر تبويب **Models** في الواجهة (إعدادات مسبقة من الكتالوج، أو رابط مخصص، أو ملف محلي). راجع [دليل الواجهة الرسومية](apps/anime_upscaler_gui/README.md) للاطّلاع على خطوات الاستخدام.

> الإصدار هو **إصدار مسبق** (`v0.2.0-preview`) وليس التزاماً بدعم الإنتاج. الحزمة غير منشورة على PyPI.

## ميزات الواجهة الرسومية

- **تكبير الصور ومقاطع الفيديو** مع قائمة انتظار دفعية: إضافة ملفات/مجلدات، إيقاف مؤقت، استئناف، إلغاء، وإعادة ترتيب.
- **معاينة مباشرة ولوحة سجلات ومراقب لوحدة معالجة الرسومات** أثناء تشغيل المهمة.
- **استدلال على CPU/CUDA** بدقة FP16، ومعالجة مقسَّمة إلى بلاطات، وحارس VRAM (تحذير / تصغير تلقائي / بلاطات / إيقاف) موجَّه لوحدات الرسوميات بذاكرة 8 غيغابايت.
- **كتالوج نماذج** يكتشف نقاط التفتيش تلقائياً في `pretrained/`، مع إعدادات مسبقة مدمجة وروابط مخصصة واستيراد محلي.
- **سمات وإمكانية وصول**: أنماط فاتحة / داكنة / عالية التباين، واختصارات لوحة مفاتيح، وواجهة إنجليزية/عربية، وحد أدنى للنافذة 800x600.
- **حفظ الإعدادات وقائمة الانتظار** خارج مجلد المصدر (مجلد بيانات التطبيق الخاص بالمستخدم).
- **تحميل آمن لنقاط التفتيش** افتراضياً؛ ويتطلب الرجوع إلى pickle موافقة صريحة.

راجع [دليل الواجهة الرسومية](apps/anime_upscaler_gui/README.md) للاختصارات والإعدادات ووصفات إضافة معماريات جديدة وتفاصيل النشر المستقل.

## النماذج المدعومة

تحتوي الواجهة الرسومية على شيفرة النماذج للأنواع التالية (وقد يعرض الكتالوج معماريات إضافية غير مدعومة لتكون ظاهرة بدل تجاهلها):

| النوع | نقاط تفتيش مثال عليها | التكبير | الرخصة |
|---|---|---:|---|
| `srvgg` | realesr-animevideov3, 4xLSDIRCompactv2 | 4x | BSD-3 / CC-BY |
| `span` | span_pix_pretrain_4x, span_mssim_pretrain_4x | 4x | CC-BY-4.0 |
| `era` | eranet_N12_pretrain_325k | 2x | MIT |
| `animesr` | AnimeSR_v1, AnimeSR_v2 | 4x | Apache-2.0 |
| `rfdn_student` | RFDN_distill_v1_4x_student | 4x | مشروع داخلي |
| `srvgg_student` | SRVGG_distill_v1_4x_student | 4x | مشروع داخلي |

`rfdn_student` و`srvgg_student` هما الطالبان الخفيفان المقطَّران داخلياً؛ وتُنزَّل نقاط تفتيش الطالبين منفصلةً (غير مرفقة). التفاصيل وملاحظات الترخيص وملف الإسناد إلى المصادر الأصلية موجودة في [دليل الواجهة الرسومية](apps/anime_upscaler_gui/README.md).

## البحث: التدريب والتقطير

تنفّذ الحزمة البحثية تدريب الاستمتاع الفائق للأنمي وتقطير المعرفة:

- معلّم **SPAN-Tiny** (شبكة الانتباه خالية المعاملات السريعة) مع خسائر مساعدة؛ ونسخة **SPAN-F** لجودة أعلى.
- **Mamba-PAN** بمسح رباعي الاتجاهات (يتطلب الحزمة الاختيارية `mamba-ssm`؛ ومسار SPAN وحده يعمل بدونها).
- **تقطير التجميع التدريجي**: تدريب عدة معلّمين، ثم دمجهم في معلّم تجميعي (meta-teacher)، ثم تقطير ذلك إلى طالب خفيف جداً.
- **MTKD + FAKD**: تقطير معرفة متعدد المعلّمين (موجَّه بالمويجات) وخسائر تقطير الارتباط المميز (feature affinity).
- **الطلاب**: RFDN (حوالي 600 ألف معامل) وTinySRVGG (حوالي 317 ألف معامل) لاستدلال سريع على CPU/CUDA.
- **خط بيانات**: تدهور بيانات على نمط RealESRGAN بإعدادات مسبقة (light/medium/heavy/anime)، وضغط على نمط APISR بمرحلتين (JPEG/WebP/AVIF/H.264/H.265)، وأوضاع معالجة مسبقة/مختلطة.
- **إضافات الاستدلال**: معالجة البلاطات، والتكبير بزيادة وقت الاختبار D4 (`--tta`)، والتصدير إلى ONNX/TorchScript، وخادم FastAPI.

التثبيت من نسخة المصدر:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"         # أدوات التطوير
pip install -e ".[api]"         # تبعيات خادم FastAPI الاختيارية
pip install -e ".[mamba]"       # دعم Mamba-PAN الاختياري (mamba-ssm)
```

أمثلة على واجهة الأوامر:

```bash
# التدريب (ملفات الإعدادات في configs/؛ راجع docs/config_reference.md)
anime-sr train --config configs/base.yaml

# الاستدلال على صورة واحدة أو مجلد
anime-sr infer --checkpoint checkpoints/model_best.pt --input ./input_images --output ./output --scale 4
anime-sr infer --checkpoint checkpoints/model_best.pt --input ./input_images --output ./output --scale 4 --tta --tile 512

# التصدير والتحقق والقياس ثم تشغيل الخادم
anime-sr export --checkpoint ./checkpoints/model_best.pt --output ./exported/model.onnx --format onnx
anime-sr validate --checkpoint ./checkpoints/model_best.pt
anime-sr benchmark --checkpoint ./checkpoints/model_best.pt --input ./input_images/sample.png --iterations 10
anime-sr serve --config configs/api/default.yaml --host 127.0.0.1 --port 8000
```

واجهة Python:

```python
from anime_sr.inference.engine import InferenceEngine

engine = InferenceEngine.from_checkpoint(
    checkpoint_path="./checkpoints/model_best.pt",
    model_type="span",      # 'span' | 'neosr_span' | 'mamba_pan' | 'tiny_student'
    device="cuda",          # أو "cpu"
)
result = engine.process_image("input.png", "output.png", scale=4, tile_size=512)
```

المزيد: [دليل التدريب](docs/training.md) · [معماريات النماذج](docs/models.md) · [مرجع الأوامر](docs/cli.md) · [الإعدادات](docs/configuration.md) · [الواجهة البرمجية](docs/api.md).

> نقاط التفتيش التجريبية المستخدمة في التوثيق ليست نماذج مُصدَرة. تدريب v7 وتقييم v8 على وحدة معالجة الرسومات مؤجَّلان إلى وحدة المستخدم، كما هو مذكور في التقارير المرتبطة.

## قياسات الأداء والنتائج المقاسة

هذه قياسات هندسية يبلّغ عنها المشروع، على جهاز واحد (Quadro RTX 4000 / RTX 4000 Mobile، Windows)، وليست مجموعة قياس مرجعية مستقلة. القيم منقولة من الوثائق المرتبطة؛ ولا تُدَّعى هنا أي قياسات جديدة.

| النموذج | المعاملات | الجودة (مُبلَّغ عنها) | الإنتاجية (مُبلَّغ عنها) | الدليل |
|---|---:|---|---|---|
| طالب RFDN (`rfdn_student`) | ‏~315K | 29.32 dB على مجموعة الاختبار (PSNR) | 2.49 إطاراً/ثانية من البداية للنهاية مع D4 TTA + FP16 من 854x480 إلى 3416x1920؛ و18.80-20.52 إطاراً/ثانية مع مسار TensorRT + NVENC (المرحلة 1) | [التقرير](docs/rfdn_realtime_report.md)، [دليل الواجهة](apps/anime_upscaler_gui/README.md) |
| طالب TinySRVGG (`srvgg_student`) | 317K | 33.55 dB / 0.9146 SSIM على مجموعة الاختبار المحجوزة (bicubic 33.05)، وNIQE 7.38 | 51.8 إطاراً/ثانية مقابل 42.3 لـ `realesr-animevideov3`، بإخراج 2560x1440 على حلقة حقيقية بمعدل 25 إطاراً/ثانية | [دليل الواجهة](apps/anime_upscaler_gui/README.md) |

- يتضمن تقرير RFDN الزمني التفصيل الكامل لكل مرحلة وتقييم TensorRT/NVENC في المرحلة 1: [docs/rfdn_realtime_report.md](docs/rfdn_realtime_report.md).
- خطوط أساس مقارنة v7 (نموذج SPAN الجاهز وضبط v4) موجودة في [docs/v7_results.md](docs/v7_results.md)؛ أما تدريب v7 نفسه فقد أُجِّل.
- شيفرة v8 لـ TTA ودمج النماذج (model souping) مُنجَزة ومختبَرة، وتقييمها على وحدة الرسوميات مؤجَّل: [docs/v8_results.md](docs/v8_results.md).
- قياسات يوم الإصدار الخام: [docs/status/MEASURED_STATUS_2026-11-16.md](docs/status/MEASURED_STATUS_2026-11-16.md).

## للباحثين

يوثّق المستودع منظومة بحثية كاملة لا نموذجاً واحداً:

- تقطير التجميع التدريجي (تجميع المعلّمين ثم الطالب)، وخسائر MTKD/FAKD، ومجموعة خسائر موجَّهة بالمويجات وإدراكية/تكاملية (adversarial)؛
- أعمدة SPAN والعمود الفقري القائم على Mamba، ومسارات استدلال ERANet وAnimeSR، وتدهور على نمط Real-ESRGAN للتدريب؛
- تحميل آمن لنقاط التفتيش (`weights_only=True` افتراضياً، مع موافقة صريحة لـ pickle)، وملفات YAML قابلة لإعادة الإنتاج، ونقاط دخول عبر واجهة الأوامر والواجهة البرمجية للتقييم.

**أعمال ذات صلة ومشاريع أصلية**

- SPAN — [شبكة الانتباه خالية المعاملات السريعة](https://github.com/hongyuanyu/SPAN) ([الورقة](https://arxiv.org/abs/2311.12770))
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) ([الورقة](https://arxiv.org/abs/2107.10833))
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [neosr](https://github.com/muslll/neosr)
- [MambaIR](https://github.com/csguoh/MambaIR) ([الورقة](https://arxiv.org/abs/2402.15648)؛ وMambaIRv2: [arXiv:2411.15269](https://arxiv.org/abs/2411.15269))
- [spandrel](https://github.com/chaiNNer-org/spandrel)
- [AnimeSR](https://github.com/TencentARC/AnimeSR) ([الورقة](https://arxiv.org/abs/2206.07038))
- [APISR](https://github.com/Kiteretsu77/APISR) ([الورقة](https://arxiv.org/abs/2403.01598))
- [Phhofm/models](https://github.com/Phhofm/models) — مصدر نقاط تفتيش SPAN الجاهزة المستخدمة كنقطة بداية للضبط.

**الاستشهاد بهذا المستودع**

- بيانات الاستشهاد: [`CITATION.cff`](CITATION.cff) (CFF 1.2.0)؛ ويعرض GitHub زر **Cite this repository** بفضلها.
- كيفية الاستشهاد بالبرمجية والأوراق البحثية الأساسية: [`docs/citation.md`](docs/citation.md).
- تُجمَع إدخالات BibTeX للأوراق الأصلية هناك؛ يُرجى أيضاً احترام رخص المصادر وملف الإسناد في [`apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md`](apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md).

## لوكلاء الذكاء الاصطناعي وأبحاث النماذج اللغوية

توجد نقاط دخول قابلة للقراءة آلياً في جذر المستودع حتى لا يحتاج الوكلاء إلى كشط موقع التوثيق:

| الملف | الغرض |
|---|---|
| [`llms.txt`](llms.txt) | فهرس موجز قابل للقراءة آلياً لهذا المستودع (وفق اصطلاح [llmstxt.org](https://llmstxt.org/)) |
| [`docs/llms.txt`](docs/llms.txt) | فهرس موقع التوثيق بروابط نسبية |
| [`AGENTS.md`](AGENTS.md) | تعليمات موجزة لوكلاء البرمجة: البنية، الأوامر، القيود |
| [`docs/AGENTS.md`](docs/AGENTS.md) | قواعد المساهمين/الوكلاء الداخلية المفصّلة المستخدمة أثناء التطوير |
| [`CITATION.cff`](CITATION.cff) | بيانات الاستشهاد (CFF 1.2.0) |
| [`docs/citation.md`](docs/citation.md) | كيفية الاستشهاد؛ وBibTeX للبرمجية والأوراق الأساسية |

نقاط دخول مستقرة:

| الواجهة | نقطة الدخول |
|---|---|
| واجهة الأوامر | `anime-sr` (`train`، `infer`، `export`، `serve`، `validate`، `benchmark`) |
| الواجهة الرسومية | `anime-upscaler-gui` (حزمة wheel)؛ ويطبع `--self-check` تقرير بيئة بصيغة JSON |
| واجهة Python | `anime_sr.inference.engine.InferenceEngine`، `anime_sr.api.inference_api.create_app` |
| السكربتات | `scripts/inference.py`، `scripts/train.py`، `scripts/release/rehearse_windows.ps1` |

مثال استدلال بدون واجهة رسومية:

```bash
pip install -e ".[dev]"
anime-sr infer --checkpoint checkpoints/model_best.pt --input ./input_images --output ./output --scale 4 --device cuda
```

أوامر التحقق:

```bash
pytest                                    # اختبارات البحث
pytest apps/anime_upscaler_gui/tests -q   # اختبارات الواجهة الرسومية
pytest tests/release -q                   # اختبارات حماية الإصدار والحزم
python -m mkdocs build --strict           # موقع التوثيق
```

ملاحظات أمان للاستخدام الآلي: تُحمَّل نقاط التفتيش بـ `weights_only=True` افتراضياً. وتقيّد الحزمة البحثية الرجوع إلى pickle خلف `security.allow_pickle_checkpoint` (متغير البيئة: `DSH_ALLOW_PICKLE_CHECKPOINT=true`)؛ وتقيّده الواجهة الرسومية خلف `ANIME_UPSCALER_ALLOW_UNSAFE_PICKLE=1`. لا تُستخدم هذه الخيارات إلا مع نقاط تفتيش موثوقة.

## الإعدادات

### متغيرات البيئة

انسخ `.env.example` إلى `.env` وعدّله؛ وتربط البادئة `DSH_*` بمسارات الإعدادات.

```bash
cp .env.example .env
```

| المتغير | القيمة الافتراضية | الوصف |
|---|---|---|
| `DSH_API_KEY` | - | مفتاح مصادقة الواجهة البرمجية |
| `DSH_MODEL_PATH` | `./checkpoints` | مسار نقاط تفتيش النماذج |
| `DSH_OUTPUT_PATH` | `./output` | مسار ملفات المخرجات |
| `DSH_LOG_LEVEL` | `INFO` | مستوى التسجيل |
| `DSH_CUDA_VISIBLE_DEVICES` | `0` | معرّفات وحدات معالجة الرسومات |
| `DSH_ALLOW_PICKLE_CHECKPOINT` | `false` | السماح بنقاط تفتيش pickle (أمان؛ للملفات الموثوقة فقط) |
| `DSH_MAX_FILE_SIZE_MB` | `50` | أقصى حجم للتحميل بالميغابايت |
| `DSH_RATE_LIMIT_PER_MIN` | `60` | حدّ معدل طلبات الواجهة البرمجية في الدقيقة |

### إعدادات التدريب

يُضبط التدريب بملفات YAML تحت `configs/`؛ راجع [docs/config_reference.md](docs/config_reference.md) و[docs/configuration.md](docs/configuration.md). تحقّق من الإعدادات بالأمر `anime-sr train --config <file> --dry-run` قبل أي تشغيل طويل.

## التطوير

```bash
pip install -e ".[dev]"
```

الاختبارات:

```bash
pytest                                  # جميع الاختبارات المحددة في pyproject.toml
pytest -m "not slow"                    # تخطي الاختبارات البطيئة
pytest -m integration                   # اختبارات التكامل فقط
pytest tests/release -q                 # اختبارات حماية الإصدار والحزم
pytest apps/anime_upscaler_gui/tests -q # اختبارات الواجهة الرسومية (تُشغَّل في CI أيضاً)
```

التوثيق:

```bash
python -m mkdocs build --strict   # docs/ -> site/ (MkDocs Material)
python -m mkdocs serve            # معاينة محلية
```

جودة الشيفرة (في CI: فحص الصياغة مانع؛ أما فحوص التنسيق والأنواع الكاملة فاستشارية):

```bash
black src/
flake8 src/
mypy src/
```

## بنية المستودع

```text
upscale_anime/
├── src/anime_sr/              # الحزمة البحثية: النماذج، التدريب، التقطير، الاستدلال، الواجهة البرمجية
├── apps/anime_upscaler_gui/   # واجهة Tkinter المستقلة + بناء حزمة wheel
├── configs/                   # إعدادات YAML (التدريب، الواجهة البرمجية، الضبط الدقيق)
├── scripts/                   # أدوات الأوامر، التقييم، تدريب الإصدار
├── tests/                     # اختبارات البحث + اختبارات tests/release
├── docs/                      # مصادر موقع التوثيق (MkDocs Material)
├── assets/                    # صور أمثلة
├── .github/workflows/         # CI: الاختبارات، الواجهة، فحص API، التكامل، الأداء، التوثيق، فحص أمان استشاري
├── pyproject.toml             # anime-sr 0.1.0 (الحزمة البحثية)
└── apps/anime_upscaler_gui/pyproject.toml  # anime-upscaler-gui 0.2.0 (حزمة wheel المُصدَرة)
```

المجلدات المولَّدة/العاملة (`data/`، `checkpoints/`، `pretrained/`، `output/`، `runs/`، `dist/`، `site/`، `.venv/`) مستثناة من git.

## التوثيق

- [فهرس التوثيق](docs/index.md)
- [دليل المستخدم](docs/USER_GUIDE.md) و[المرجع السريع](docs/QUICK_REFERENCE.md)
- [مرجع واجهة الأوامر](docs/cli.md) و[الإعدادات](docs/configuration.md)
- [معماريات النماذج](docs/models.md) و[نظرة عامة على المعمارية](docs/architecture.md)
- [دليل التدريب](docs/training.md)
- [النشر](docs/deployment.md) و[الواجهة البرمجية العامة](docs/PUBLIC_API.md)
- [استكشاف الأخطاء وإصلاحها](docs/troubleshooting.md)
- [دليل الاستشهاد](docs/citation.md) و[llms.txt](llms.txt) للاستهلاك الآلي
- [دليل الواجهة الرسومية](apps/anime_upscaler_gui/README.md)

## الحالة والقيود

- **إصدار تجريبي مسبق فقط.** الواجهة الرسومية إصدار مسبق `v0.2.0-preview`، والحزمة البحثية ألفا. لا شيء هنا مدعوم للإنتاج.
- **المنصات.** بُني إصدار الواجهة الرسومية واختُبر على **Windows 11**؛ ويعمل CI على Ubuntu Linux، لذا لا تُعتمَد المنصات المكتبية الأخرى للحزمة الجاهزة.
- **لا حزمة على PyPI.** تُوزَّع حزمة الواجهة الرسومية عبر إصدار GitHub فقط. ومهمة `publish-pypi` في CI يدوية فقط (ولم تُشغَّل قط)، ولا تُنشر أي حزمة على PyPI.
- **لا أوزان مرفقة.** تُنزَّل نقاط تفتيش النماذج أو تُستورَد منفصلةً؛ ونقاط التفتيش التجريبية المذكورة في التوثيق ليست نماذج مُصدَرة.
- **مسار Mamba اختياري.** تتطلب Mamba-PAN الحزمة الاختيارية `mamba-ssm`؛ أما مسار SPAN فلا يتطلبها. ويُختبَر مسار v7 الاختياري وإعداد Windows بدون نواة CUDA ضمن مجموعة الاختبارات.
- **فحص الأمان استشاري.** مهمة bandit في CI غير مانعة وتُبلِّغ حالياً عن 40 نتيجة MEDIUM موجودة مسبقاً (39 موضع `torch.load` قديم و1 `urlopen`)، وهي متابَعة كعمل لاحق. تُحمَّل نقاط التفتيش بأمان افتراضياً؛ ويتطلب تحميل pickle موافقة صريحة.
- **الأرقام المُبلَّغ عنها من جهاز واحد.** القياسات في التوثيق قياسات يبلّغ عنها المشروع، وليست نتائج مستقلة أُعيد إنتاجها.
- **أعمال مؤجَّلة.** تدريب v7 وتقييم v8 مؤجَّلان إلى وحدة رسوميات المستخدم؛ ويحتوي كتالوج الواجهة العربية على 35 نصاً مترجماً يدوياً بانتظار مراجعة متحدث أصلي.
- **متابعات المسار الثاني (Track 2)** (تقوية مسار الإصدار، وإيقاف مسار الإصدار القديم، وتوسعة اختبار المنصات) متابَعة داخلياً؛ ومسار الإصدار قابل لإعادة الإنتاج عبر `python -m build` + `twine check` وسكربت التدريب.

## الرخصة

رخصة MIT - راجع [LICENSE](LICENSE).

## شكر وتقدير

- [SPAN](https://github.com/hongyuanyu/SPAN) لمعمارية الانتباه خالية المعاملات؛ و[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) و[BasicSR](https://github.com/XPixelGroup/BasicSR) و[neosr](https://github.com/muslll/neosr) لأدوات التدريب والتدهور.
- [MambaIR](https://github.com/csguoh/MambaIR) / MambaIRv2 لكتل الاستعادة القائمة على فضاء الحالة؛ و[spandrel](https://github.com/chaiNNer-org/spandrel) للتوافق بين النماذج ونقاط التفتيش.
- [Phhofm/models](https://github.com/Phhofm/models) لنقاط تفتيش SPAN الجاهزة؛ و[AnimeSR](https://github.com/TencentARC/AnimeSR) و[APISR](https://github.com/Kiteretsu77/APISR) كمراجع لاستعادة الأنمي.
- إسناد شيفرة النماذج المضمَّنة: [`apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md`](apps/anime_upscaler_gui/anime_upscaler_gui/_models/NOTICE.md).

</div>
