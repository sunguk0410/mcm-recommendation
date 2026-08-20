FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cache the rembg model in the image so application startup never downloads it.
ENV U2NET_HOME=/root/.u2net
RUN python -c "from rembg import new_session; new_session('u2net')"

COPY src ./src
COPY checkpoints/recrec_v2_best.pt ./checkpoints/recrec_v2_best.pt
COPY MCM_제품리스트_통합_추천모델용.xlsx .

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
