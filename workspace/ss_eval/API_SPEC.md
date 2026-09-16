# Document AI - Parsing API 명세서

## 개요

| 항목 | 내용 |
|------|------|
| 서버 주소 | `http://211.109.9.61:30720` |
| Base URL | `/v1` |
| Content-Type | `multipart/form-data` (파일 업로드), `application/json` (조회) |

---

## API 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/v1/document-digitization` | 동기 추론 (즉시 결과 반환) |
| POST | `/v1/jobs/submit` | 비동기 작업 제출 |
| GET | `/v1/jobs/{uuid}` | 작업 상태 조회 |
| GET | `/v1/jobs` | 전체 작업 목록 |
| POST | `/v1/jobs/{uuid}/reparse` | 기존 파일로 재파싱 |
| GET | `/v1/results/{uuid}` | UUID로 결과 조회 (파일 시스템 기반) |
| GET | `/v1/files/{uuid}/{filename}` | 원본 파일 다운로드 |
| GET | `/v1/pdf-base64/{uuid}` | PDF를 Base64로 변환 |
| GET | `/v1/images/{uuid}/{image_name}` | 이미지 조회 |

---

## 1. 동기 추론 API

문서를 업로드하면 즉시 파싱 결과를 반환합니다.

### Request

```http
POST /v1/document-digitization
Content-Type: multipart/form-data
```

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| file | File | O | - | 업로드할 문서/이미지 파일 |
| use_chart_recognition | boolean | X | true | 차트/그래프 인식 여부 |
| markdown_ignore_labels | string | X | null | 마크다운에서 제외할 라벨 (쉼표 구분) |
| use_base64_images | boolean | X | false | 이미지 base64 변환 여부 |

### Response (성공)

```json
{
  "status": 200,
  "elements": [
    {
      "page": 1,
      "markdown": "# 제목\n\n본문 내용...",
      "html": "<h1>제목</h1><p>본문 내용...</p>",
      "json": {...},
      "images": {
        "page1_image_1": "data:image/png;base64,..."
      }
    }
  ],
  "md": "전체 마크다운 합본",
  "html": "전체 HTML 합본",
  "images": {
    "page1_image_1": "data:image/png;base64,..."
  }
}
```

### Response (실패)

```json
{
  "status": 400,
  "error": "에러 메시지"
}
```

---

## 2. 비동기 작업 제출 API

문서를 업로드하고 job_id를 즉시 반환합니다. 실제 처리는 백그라운드에서 진행됩니다.

### Request

```http
POST /v1/jobs/submit
Content-Type: multipart/form-data
```

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| file | File | O | - | 업로드할 문서/이미지 파일 |
| use_chart_recognition | boolean | X | true | 차트/그래프 인식 여부 |
| markdown_ignore_labels | string | X | null | 마크다운에서 제외할 라벨 (쉼표 구분) |
| use_base64_images | boolean | X | false | 이미지 base64 변환 여부 |

### Response

```json
{
  "status": 202,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Job submitted successfully"
}
```

---

## 3. 재파싱 API

기존 파싱 결과의 원본 파일을 재사용하여 새 옵션으로 재파싱합니다.

### Request

```http
POST /v1/jobs/{uuid}/reparse
Content-Type: multipart/form-data
```

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| uuid | string | O | - | 기존 작업 UUID |
| use_chart_recognition | boolean | X | true | 차트/그래프 인식 여부 |
| markdown_ignore_labels | string | X | null | 마크다운에서 제외할 라벨 (쉼표 구분) |
| use_base64_images | boolean | X | false | 이미지 base64 변환 여부 |

### Response (성공)

```json
{
  "status": 202,
  "job_id": "new-uuid-for-reparse-job",
  "message": "Reparse job submitted successfully"
}
```

### Response (실패)

```json
{
  "status": 404,
  "error": "Original job not found"
}
```

---

## 4. 작업 상태 조회 API

UUID로 작업 상태를 조회합니다. 결과 데이터는 포함되지 않으며, 완료된 작업의 결과는 `/v1/results/{uuid}`를 사용하세요.

### Request

```http
GET /v1/jobs/{uuid}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| uuid | string | O | 작업 UUID |

### Response (대기 중)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "job_status": "pending",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00",
  "queue_position": 2
}
```

### Response (처리 중)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "job_status": "processing",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00"
}
```

### Response (완료)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "job_status": "completed",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:31:00"
}
```

> **Note**: 완료된 작업의 결과는 `/v1/results/{uuid}` API를 통해 조회하세요.
```

### Response (실패)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "job_status": "failed",
  "filename": "document.pdf",
  "created_at": "2024-01-15T10:30:00",
  "completed_at": "2024-01-15T10:31:00",
  "error": "에러 메시지"
}
```

### 작업 상태 (job_status)

| 상태 | 설명 |
|------|------|
| pending | 대기 중 |
| processing | 처리 중 |
| completed | 완료 |
| failed | 실패 |

---

## 5. 전체 작업 목록 API

모든 작업 목록을 조회합니다.

### Request

```http
GET /v1/jobs
```

### Response

```json
{
  "status": 200,
  "total": 10,
  "jobs": [
    {
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "filename": "document.pdf",
      "status": "completed",
      "created_at": "2024-01-15T10:30:00"
    },
    ...
  ]
}
```

---

## 6. UUID로 결과 조회 API (파일 시스템 기반)

UUID로 파싱 결과를 조회합니다. 서버 재시작 후에도 조회 가능합니다.

### Request

```http
GET /v1/results/{uuid}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| uuid | string | O | 작업 UUID |

### Response (성공)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "uuid": "550e8400-e29b-41d4-a716-446655440000",
    "status": 200,
    "elements": [...],
    "md": "마크다운 내용",
    "html": "HTML 내용",
    "images": {...}
  },
  "previewFile": "/v1/files/550e8400-.../document.pdf",
  "previewFilename": "document.pdf",
  "originalFile": "/v1/files/550e8400-.../document.pptx",
  "originalFilename": "document.pptx",
  "files": ["document.pptx", "document.pdf"]
}
```

### Response (실패)

```json
{
  "status": 404,
  "error": "Result not found"
}
```

### 파일 정보 설명

| 필드 | 설명 |
|------|------|
| previewFile | 미리보기용 파일 URL (PDF 우선) |
| previewFilename | 미리보기 파일명 |
| originalFile | 원본 파일 URL |
| originalFilename | 원본 파일명 |
| files | 폴더 내 전체 파일 목록 |

---

## 7. 원본 파일 다운로드 API

UUID와 파일명으로 원본 파일을 다운로드합니다.

### Request

```http
GET /v1/files/{uuid}/{filename}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| uuid | string | O | 작업 UUID |
| filename | string | O | 파일명 |

### Response

파일 바이너리 (Content-Type은 파일 확장자에 따라 결정)

---

## 8. PDF Base64 변환 API

UUID로 PDF 파일을 찾아 Base64로 인코딩하여 반환합니다.

### Request

```http
GET /v1/pdf-base64/{uuid}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| uuid | string | O | 작업 UUID |

### Response (성공)

```json
{
  "status": 200,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "document.pdf",
  "pdfBase64": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8P...",
  "pdfDataUri": "data:application/pdf;base64,JVBERi0xLjQKJeLjz9MKMSAwIG9iago8P...",
  "size": 123456
}
```

| 필드 | 설명 |
|------|------|
| pdfBase64 | Base64로 인코딩된 PDF 데이터 |
| pdfDataUri | Data URI 형식 (바로 사용 가능) |

### Response (실패)

```json
{
  "status": 404,
  "error": "PDF file not found"
}
```

### 사용 예시

```bash
curl "http://211.109.9.61:30720/v1/pdf-base64/550e8400-e29b-41d4-a716-446655440000"
```

---

## 9. 이미지 조회 API

파싱 결과의 이미지를 조회합니다. (use_base64_images=false인 경우 사용)

### Request

```http
GET /v1/images/{uuid}/{image_name}
```

| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| uuid | string | O | 작업 UUID |
| image_name | string | O | 이미지 파일명 (예: page1_image_1.png) |

### Response

이미지 바이너리 (Content-Type: image/png 등)

---

## 지원 파일 형식

| 분류 | 확장자 |
|------|--------|
| 이미지 | jpg, jpeg, png, gif, webp, bmp |
| 문서 | pdf, doc, docx |
| 프레젠테이션 | ppt, pptx |
| 스프레드시트 | csv, xls, xlsx, xlsm, xlsb, ods |
| 구조화 데이터 | json |

---

## 사용 예시

### cURL - 비동기 작업 제출

```bash
curl -X POST "http://211.109.9.61:30720/v1/jobs/submit" \
  -F "file=@document.pdf" \
  -F "use_chart_recognition=true" \
  -F "use_base64_images=true"
```

### cURL - 작업 상태 조회

```bash
curl "http://211.109.9.61:30720/v1/jobs/550e8400-e29b-41d4-a716-446655440000"
```

### cURL - UUID로 결과 조회

```bash
curl "http://211.109.9.61:30720/v1/results/550e8400-e29b-41d4-a716-446655440000"
```

---

## 결과 뷰어 URL

파싱 결과를 웹 UI로 확인할 수 있습니다.

```
https://{도메인}/document-ai/results/{uuid}
```

**예시:**
```
http://211.109.9.61:60303/document-ai/results/550e8400-e29b-41d4-a716-446655440000
```

---

## 연동 흐름

```
1. 문서 업로드
   POST /v1/jobs/submit
   ↓
2. uuid 저장
   ↓
3. 상태 폴링
   GET /v1/jobs/{uuid}
   (job_status가 completed/failed가 될 때까지)
   ↓
4. 결과 조회
   GET /v1/results/{uuid}
   ↓
5. 결과 사용
   - API: md, html, elements 사용
   - 뷰어: /document-ai/results/{uuid} URL로 이동
```

---

## 에러 코드

| HTTP Status | 설명 |
|-------------|------|
| 200 | 성공 |
| 202 | 작업 제출 성공 (비동기) |
| 400 | 잘못된 요청 |
| 404 | 리소스 없음 |
| 500 | 서버 내부 오류 |
