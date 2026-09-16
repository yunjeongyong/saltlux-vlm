"""facts.py 하나에서 엑셀 3종을 생성한다. 수치는 facts.py 에만 있다."""
import sys
sys.path.insert(0,"/data/workspace/yyj/data/reports/build")
import facts as F
import sources as SRC
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

TH=Side(style="thin",color="BFBFBF"); BD=Border(left=TH,right=TH,top=TH,bottom=TH)
HF=PatternFill("solid",fgColor="D9E2F3"); GF=PatternFill("solid",fgColor="B4C7E7")
NEW=PatternFill("solid",fgColor="FFF2CC"); HI=PatternFill("solid",fgColor="E2EFDA")
OK=PatternFill("solid",fgColor="E2EFDA"); NG=PatternFill("solid",fgColor="FCE4D6")
GOALF=PatternFill("solid",fgColor="FCE4D6"); GRY=PatternFill("solid",fgColor="F2F2F2")
C=Alignment(horizontal="center",vertical="center",wrap_text=True)
L=Alignment(horizontal="left",vertical="center",wrap_text=True)
ROOT="/data/workspace/yyj/data"

def W(ws,w):
    for i,x in enumerate(w,1): ws.column_dimensions[get_column_letter(i)].width=x
def h(ws,r,c,v,fill=HF):
    x=ws.cell(r,c,v); x.font=Font(bold=True,size=10); x.fill=fill; x.border=BD; x.alignment=C
def cl(ws,r,c,v,fill=None,align=None,bold=False,fmt=None):
    x=ws.cell(r,c,v); x.border=BD; x.alignment=align or C
    if fill: x.fill=fill
    if bold: x.font=Font(bold=True,size=10)
    if fmt: x.number_format=fmt
def legend(ws,r,items):
    """색이 무슨 뜻인지 표 아래에 적어 둔다. 시트마다 색의 의미가 달라 범례가 필요하다."""
    x=ws.cell(r,2,"범례"); x.font=Font(bold=True,size=9); x.alignment=C
    c=3
    for fill,label in items:
        a=ws.cell(r,c,""); a.fill=fill; a.border=BD
        b=ws.cell(r,c+1,label); b.alignment=L; b.font=Font(size=9)
        c+=2
    return r+1

def note(ws,r,c2,txt):
    cl(ws,r,2,txt,align=L); ws.merge_cells(start_row=r,start_column=2,end_row=r,end_column=c2)

# ══════════════ 01 데이터셋 카탈로그 ══════════════
def mk01():
    wb=Workbook(); ws=wb.active; ws.title="1.원본 데이터셋"
    W(ws,[3,6,32,26,20,20,11,10,10,9,9,9,52,34])
    cl(ws,2,2,"수집한 원본 데이터 전체   (수량은 2026-09-01 디스크 실측)",bold=True,align=L)
    ws.merge_cells("B2:N2")
    h(ws,3,2,"ID",GF); ws.merge_cells("B3:B4")
    h(ws,3,3,"데이터명",GF); ws.merge_cells("C3:C4")
    h(ws,3,4,"텍소노미",GF); ws.merge_cells("D3:F3")
    for i,n in enumerate(["태스크","도메인","수집 방법"],4): h(ws,4,i,n)
    h(ws,3,7,"수량",GF); ws.merge_cells("G3:I3")
    h(ws,4,7,"공개 규모"); h(ws,4,8,"보유"); h(ws,4,9,"활용")
    h(ws,3,10,"언어 (활용 행 기준)",GF); ws.merge_cells("J3:L3")
    for i,n in enumerate(["KOR","라틴문자","기타"],10): h(ws,4,i,n)
    h(ws,3,13,"경로",GF); ws.merge_cells("M3:M4")
    h(ws,3,14,"출처 / 라이선스",GF); ws.merge_cells("N3:N4")
    ws.freeze_panes="C5"
    FILL={"new":NEW,"drop":GRY,"val":PatternFill("solid",fgColor="DDEBF7"),"use":None}
    r=5; tp=tb=tu=tk=tl_=te=0
    for sid,nm,task,dom,how,pub,have,path,lic,st in SRC.SOURCES:
        f=FILL[st]; u=SRC.used(sid)
        lg=SRC.LANG.get(sid,{})
        ko=lg.get("KOR",0); la=lg.get("라틴문자",0)
        et=sum(v for k,v in lg.items() if k not in ("KOR","라틴문자"))
        row=[sid,nm,task,dom,how,pub if pub else "-",have,u if u else "-",
             ko or "-",la or "-",et or "-",path,lic]
        for i,v in enumerate(row,2):
            cl(ws,r,i,v,fill=f,align=L if i in (3,4,5,6,13,14) else C,
               fmt="#,##0" if isinstance(v,int) else None)
        if pub: tp+=pub
        tb+=have; tu+=u; tk+=ko; tl_+=la; te+=et; r+=1
    cl(ws,r,3,"합계",bold=True)
    for i,v in zip((7,8,9,10,11,12),(tp,tb,tu,tk,tl_,te)): cl(ws,r,i,v,bold=True,fmt="#,##0")
    for i in (2,4,5,6,13,14): cl(ws,r,i,"")
    r+=2
    r=legend(ws,r,[(NEW,"exp_012 신규 투입"),(PatternFill("solid",fgColor="DDEBF7"),"검증 전용"),
                   (GRY,"미활용 (폐기·라이선스 미확인)")])+0
    note(ws,r,14,"※ 수량 3종 구분 — 공개 규모: 외부 데이터셋이 배포하는 원 규모(참고용, 보유량이 아님) / "
                 "보유: 우리 디스크의 실제 파일 수 / 활용: exp_012 학습셋에 들어간 고유 이미지 수. "
                 "종전에는 셋이 한 칸에 섞여 있어 추적이 되지 않았다. "
                 "언어는 활용 행의 정답 텍스트에서 HTML 태그를 걷어내고 문자 종류로 판정한 값이다."); r+=2
    note(ws,r,14,"※ 행 색 — 노랑: exp_012 신규 투입 / 파랑: 검증 전용 / 회색: 미활용(폐기·라이선스 미확인). "
                 "S09~S12 S16 은 별도 폴더가 아니라 cand_crops 안에 접두어로 구분되어 있다. "
                 "경로 기준 /data/workspace/yyj/data/")

    ws=wb.create_sheet("2.학습 활용 데이터"); W(ws,[3,20,20,10,8,10,40,46,10])
    cl(ws,2,2,"학습에 실제로 사용한 데이터 — 버전별",bold=True,align=L); ws.merge_cells("B2:I2")
    h(ws,3,2,"학습셋 버전",GF); ws.merge_cells("B3:B4")
    h(ws,3,3,"사용 실험",GF); ws.merge_cells("C3:C4")
    h(ws,3,4,"규모",GF); ws.merge_cells("D3:F3")
    h(ws,4,4,"train 행"); h(ws,4,5,"val 행"); h(ws,4,6,"고유 이미지")
    h(ws,3,7,"원본 매핑 (S-ID)",GF); ws.merge_cells("G3:G4")
    h(ws,3,8,"경로",GF); ws.merge_cells("H3:H4")
    h(ws,3,9,"상태",GF); ws.merge_cells("I3:I4")
    ws.freeze_panes="C5"
    r=5
    for v,e,tr,va,path,stt in SRC.VERSIONS:
        m=SRC.VER_SRC[v]; f=NEW if v=="exp012_mix" else (HI if v=="exp011_mix" else None)
        mp=" ".join(sorted(m))          # 장수는 '2-1.원본×버전 매핑' 시트에서 본다
        for i,x in enumerate([v,e,tr,va,sum(m.values()),mp,path,stt],2):
            cl(ws,r,i,x,fill=f,align=L if i in (2,3,7,8) else C,
               fmt="#,##0" if isinstance(x,int) else None)
        r+=1
    r+=1
    r=legend(ws,r,[(NEW,"exp_012 (진행 중)"),(HI,"exp_011 (현 최고 성능)")])
    note(ws,r,9,"※ 원본 매핑의 S-ID 는 1.원본 데이터셋 시트와 대응한다. 출처별로 몇 장을 썼는지는 "
                "다음 시트 '2-1.원본×버전 매핑' 에서 확인한다. 예: exp012_mix 는 exp011_mix 에 "
                "S17 S18 S19 S20 을 더한 것이다."); r+=2
    cl(ws,r,2,"태스크별 구성 — 버전 비교 (행 수)",bold=True,align=L)
    ws.merge_cells(start_row=r,start_column=2,end_row=r,end_column=9); r+=1
    for i,n in enumerate(["task","exp_004","exp_009","exp_010","exp_011","exp_012","원본 S-ID"],2): h(ws,r,i,n)
    ws.merge_cells(start_row=r,start_column=8,end_row=r,end_column=9); ws.cell(r,9).fill=HF; ws.cell(r,9).border=BD
    r+=1
    T=[("receipt_crop_text",820,34391,820,12000,12000,"S01 S02 S07 S09"),
       ("receipt_markdown",182,39866,4368,6196,6196,"S03 S04 S05"),
       ("html_table",5461,5461,"-",5461,5461,"S14"),
       ("doc_parsing/invoice","-","-","-","-",4491,"S19 S20"),
       ("parsing/전체텍스트",2499,2499,2499,2499,2499,"S25"),
       ("page_ocr",2000,2000,2000,2000,2000,"S13"),
       ("receipt_crop_table",72,1900,72,1900,1900,"S01 S07"),
       ("doc_parsing/certificate_of_origin","-","-","-","-",1127,"S17 S18"),
       ("doc_parsing/bill_of_lading","-","-","-","-",1125,"S17 S18"),
       ("doc_parsing/securities_acquisition","-","-","-","-",1125,"S17 S18"),
       ("doc_parsing/change_registration","-","-","-","-",1104,"S17 S18"),
       ("doc_parsing/insurance_policy","-","-","-","-",1100,"S17 S18"),
       ("doc_parsing/random_grid","-","-","-","-",1096,"S17 S18"),
       ("doc_parsing/packing_list","-","-","-","-",1091,"S17 S18"),
       ("doc_parsing/commercial_invoice","-","-","-","-",1091,"S17 S18"),
       ("doc_parsing/power_of_attorney","-","-","-","-",1082,"S17 S18"),
       ("doc_parsing/certificate_of_quality","-","-","-","-",1058,"S17 S18"),
       ("doc_parsing/customer_verification","-","-","-","-",1005,"S17 S18"),
       ("doc_parsing/disclosure_request","-","-","-","-",996,"S17 S18"),
       ("doc_parsing/government",106,106,106,106,106,"S15"),
       ("doc_parsing/paper",33,33,33,33,33,"S15")]
    for row in T:
        for i,v in enumerate(row,2):
            cl(ws,r,i,v,fill=NEW if (i==7 and v!="-") else None,
               align=L if i in (2,8) else C,fmt="#,##0" if isinstance(v,int) else None)
        ws.merge_cells(start_row=r,start_column=8,end_row=r,end_column=9); ws.cell(r,9).border=BD
        r+=1
    cl(ws,r,2,"합계",bold=True)
    for i,v in enumerate([11173,86256,9898,30195,47686],3): cl(ws,r,i,v,bold=True,fmt="#,##0")
    cl(ws,r,8,""); ws.merge_cells(start_row=r,start_column=8,end_row=r,end_column=9); ws.cell(r,9).border=BD

    ws=wb.create_sheet("2-1.원본×버전 매핑")
    VN=[v for v,*_ in SRC.VERSIONS]
    W(ws,[3,7,32,20]+[13]*len(VN)+[11])
    cl(ws,2,2,"어느 원본을 어느 학습셋에 몇 장 썼는가 — 고유 이미지 수",bold=True,align=L)
    ws.merge_cells(start_row=2,start_column=2,end_row=2,end_column=5+len(VN))
    h(ws,3,2,"ID",GF); h(ws,3,3,"데이터명",GF); h(ws,3,4,"도메인",GF)
    for i,v in enumerate(VN,5): h(ws,3,i,v)
    h(ws,3,5+len(VN),"보유",GF)
    ws.freeze_panes="E4"
    r=4
    for sid,nm,task,dom,how,pub,have,path,lic,st in SRC.SOURCES:
        used_any=any(SRC.VER_SRC[v].get(sid) for v in VN)
        f=NEW if st=="new" else (PatternFill("solid",fgColor="DDEBF7") if st=="val"
                                 else (GRY if not used_any else None))
        cl(ws,r,2,sid,fill=f); cl(ws,r,3,nm,fill=f,align=L); cl(ws,r,4,dom,fill=f,align=L)
        for i,v in enumerate(VN,5):
            x=SRC.VER_SRC[v].get(sid)
            cl(ws,r,i,x if x else "-",fill=(HI if x and v=="exp012_mix" else f),
               fmt="#,##0" if x else None)
        cl(ws,r,5+len(VN),have,fill=f,fmt="#,##0"); r+=1
    cl(ws,r,3,"고유 이미지 합계",bold=True,align=L); cl(ws,r,2,""); cl(ws,r,4,"")
    for i,v in enumerate(VN,5):
        cl(ws,r,i,sum(SRC.VER_SRC[v].values()),bold=True,fmt="#,##0")
    cl(ws,r,5+len(VN),sum(x[6] for x in SRC.SOURCES),bold=True,fmt="#,##0"); r+=1
    cl(ws,r,3,"학습 행 수",bold=True,align=L); cl(ws,r,2,""); cl(ws,r,4,"")
    TR={v:tr for v,e,tr,va,path,stt in SRC.VERSIONS}
    for i,v in enumerate(VN,5): cl(ws,r,i,TR[v],bold=True,fmt="#,##0")
    cl(ws,r,5+len(VN),""); r+=2
    r=legend(ws,r,[(NEW,"exp_012 신규 투입"),(HI,"exp_012 사용분"),
                   (PatternFill("solid",fgColor="DDEBF7"),"검증 전용"),(GRY,"미활용")])
    note(ws,r,5+len(VN),
      "※ 각 버전 train.jsonl 의 이미지 경로를 역추적해 센 고유 이미지 수다. 한 이미지를 여러 행으로 "
      "학습하는 경우가 있어 '학습 행 수' 가 더 크다(예: exp009_full 은 48,400장을 86,256행으로 반복 학습). "
      "가로로 읽으면 그 원본이 어느 실험에 쓰였는지, 세로로 읽으면 그 실험이 어느 원본으로 구성됐는지 보인다.")

    ws=wb.create_sheet("3.평가 데이터"); W(ws,[3,22,10,18,12,8,16,28,44,40])
    cl(ws,2,2,"평가·검증에 사용하는 데이터 (학습에 쓰지 않음)",bold=True,align=L); ws.merge_cells("B2:J2")
    for i,n in enumerate(["이름","역할","태스크","단위","수량","지표","정답 출처","경로","비고"],2): h(ws,3,i,n)
    ws.freeze_panes="C4"
    E=[("eval_crops (text)","평가셋","OCR","블록 크롭",921,"CER","자체 검수 (S03 92장)","vlm_exp34/receipt_data/eval_crops","학습 방향 판단용",None),
       ("eval_crops (table)","평가셋","TSR","블록 크롭",98,"TEDS","자체 검수 (S03 92장)","vlm_exp34/receipt_data/eval_crops","표 구조 평가",None),
       ("SS receipt","평가셋","KIE","문서 전체",100,"KIE 필드 정확도","GPT-5.6 검증 (강한빛 선임)","ss_eval/data/receipt · ss_eval/kie_dataset/receipt","납품 판정용",None),
       ("SS quotation (견적서)","평가셋","KIE","문서 전체",100,"KIE 필드 정확도","GPT-5.6 검증 (강한빛 선임)","ss_eval/data/quotation · ss_eval/kie_dataset/quotation","대응 학습 데이터 없음",None),
       ("SS invoice (계산서)","평가셋","KIE","문서 전체",100,"KIE 필드 정확도","GPT-5.6 검증 (강한빛 선임)","ss_eval/data/invoice · ss_eval/kie_dataset/invoice","절대값 최저",None),
       ("SS bill (명세서)","평가셋","KIE","문서 전체",100,"KIE 필드 정확도","GPT-5.6 검증 (강한빛 선임)","ss_eval/data/bill · ss_eval/kie_dataset/bill","학습 후 유일 하락",None),
       ("exp val (구 검증셋)","검증셋","Document Parsing","페이지 전체",90,"eval_loss","자체 검수 (S03)","vlm_exp34/receipt_data/exp012_mix/val.jsonl","영수증 100% · 대표성 부족으로 대체",GRY),
       ("val_4dom (검증셋)","검증셋","Document Parsing","페이지 전체",240,"eval_loss","자체 검수 60 (S03) + 생성 정답 180 (S24)","vlm_exp34/receipt_data/val_4dom/val.jsonl","도메인 4종 각 60행 · exp_013 부터 적용",NEW)]
    r=4
    for row in E:
        f=row[9]
        for i,v in enumerate(row[:9],2):
            cl(ws,r,i,v,fill=f,align=L if i in (2,8,9,10) else C,fmt="#,##0" if i==6 else None)
        r+=1
    r+=1
    r=legend(ws,r,[(NEW,"신규 구축"),(GRY,"대체되어 미사용")])
    note(ws,r,10,"※ 검증셋과 평가셋은 역할이 다르다. 검증셋은 학습 도중 상태 확인과 체크포인트 선택에 쓰고, "
                 "평가셋은 학습 종료 후 판정에 쓴다. 평가셋을 체크포인트 선택에 쓰면 보고 점수가 실제 배포 "
                 "성능보다 높아지므로 사용하지 않는다."); r+=2
    note(ws,r,10,"※ 학습 데이터와 겹치지 않는다. SS 400건은 md5 대조로 학습 이미지와 겹침 0건. "
                 "val_4dom 은 학습에 쓰지 않은 시드(90001)로 새로 렌더링해 이미지 경로·정답 내용 모두 겹침 0건 확인. "
                 "GPT 검증 라벨은 정답지이므로 학습에 투입하지 않는다.")

    ws=wb.create_sheet("4.텍소노미"); W(ws,[3,26,8,44,20,18,34])
    cl(ws,2,2,"표준 용어 정의 — 태스크 구분",bold=True,align=L); ws.merge_cells("B2:G2")
    for i,n in enumerate(["표준 용어","약어","정의","출력 형태","평가 지표","현재 태스크명"],2): h(ws,3,i,n)
    X=[("Text Recognition","OCR","이미지에 보이는 글자를 그대로 전사. 구조는 만들지 않는다.","평문 텍스트","CER","page_ocr / receipt_crop_text"),
       ("Document Parsing","DP","페이지 전체를 읽기 순서대로 구조화된 문서로 만든다.","마크다운 + HTML 표","CER / TEDS","receipt_markdown / parsing/전체텍스트 / doc_parsing/*"),
       ("Table Structure Recognition","TSR","표 영역을 행·열·병합 구조가 보존된 HTML 로 만든다.","HTML <table>","TEDS","html_table / receipt_crop_table"),
       ("Key Information Extraction","KIE","정해진 스키마의 필드 값만 뽑아낸다.","JSON","필드 정확도 / F1","(SS 평가에서만 사용)"),
       ("Layout Analysis","-","블록의 종류와 읽기 순서를 예측한다.","블록 라벨 + 순서","-","(학습 없음 · 라벨 보유)"),
       ("Grounding / Detection","-","텍스트나 필드의 위치(bbox)를 예측한다.","bbox 좌표","IoU / mAP","(학습 없음 · S10 보유)")]
    r=4
    for row in X:
        for i,v in enumerate(row,2): cl(ws,r,i,v,align=C if i==3 else L)
        r+=1
    r+=2
    cl(ws,r,2,"수집 방법 분류",bold=True,align=L); ws.merge_cells(start_row=r,start_column=2,end_row=r,end_column=7); r+=1
    h(ws,r,2,"수집 방법"); h(ws,r,3,"설명")
    ws.merge_cells(start_row=r,start_column=3,end_row=r,end_column=5)
    for c in (4,5): ws.cell(r,c).border=BD; ws.cell(r,c).fill=HF
    h(ws,r,6,"해당 S-ID"); ws.merge_cells(start_row=r,start_column=6,end_row=r,end_column=7)
    ws.cell(r,7).border=BD; ws.cell(r,7).fill=HF; r+=1
    import collections as _c
    g=_c.defaultdict(list)
    for sid,nm,task,dom,how,pub,have,path,lic,st in SRC.SOURCES:
        k=("생성형 (렌더링)" if how.startswith("생성형") else how)
        g[k].append(sid)
    DESC={"사내 수집":"직접 촬영·수집 후 자체 라벨링","공개 데이터":"외부 공개 데이터셋 사용",
          "생성형 (렌더링)":"템플릿·엔진으로 문서를 만들고 정답을 자동 생성","크롤링 (웹 수집)":"웹에서 수집 후 검수"}
    for k in ["사내 수집","공개 데이터","생성형 (렌더링)","크롤링 (웹 수집)"]:
        cl(ws,r,2,k,align=L); cl(ws,r,3,DESC[k],align=L)
        ws.merge_cells(start_row=r,start_column=3,end_row=r,end_column=5)
        for c in (4,5): ws.cell(r,c).border=BD
        cl(ws,r,6," ".join(g[k]),align=L)
        ws.merge_cells(start_row=r,start_column=6,end_row=r,end_column=7); ws.cell(r,7).border=BD
        r+=1
    ws=wb.create_sheet("5.데이터 특성")
    NC=len(SRC.RES_COLS)
    W(ws,[3,16]+[13]*NC+[11,12])
    cl(ws,2,2,"① 이미지 해상도 분포 — exp_012 학습셋 47,686행",bold=True,align=L)
    ws.merge_cells(start_row=2,start_column=2,end_row=2,end_column=4+NC)
    h(ws,3,2,"해상도",GF)
    for i,c in enumerate(SRC.RES_COLS,3): h(ws,3,i,c)
    h(ws,3,3+NC,"합계",GF); h(ws,3,4+NC,"비율",GF)
    ws.freeze_panes="C4"
    r=4; n=sum(sum(SRC.RES[c]) for c in SRC.RES_COLS)
    for bi,lab in enumerate(SRC.RES_LABELS):
        over = bi>=4
        cl(ws,r,2,lab,align=L,fill=NG if over else None)
        s=0
        for i,c in enumerate(SRC.RES_COLS,3):
            v=SRC.RES[c][bi]; s+=v
            cl(ws,r,i,v,fmt="#,##0",fill=NG if over else (NEW if c=="신규 서식" else None))
        cl(ws,r,3+NC,s,fmt="#,##0",bold=True,fill=NG if over else None)
        cl(ws,r,4+NC,s/n,fmt="0.0%",fill=NG if over else None); r+=1
    cl(ws,r,2,"합계",bold=True); tot=0
    for i,c in enumerate(SRC.RES_COLS,3):
        s=sum(SRC.RES[c]); tot+=s
        cl(ws,r,i,s,fmt="#,##0",bold=True,fill=NEW if c=="신규 서식" else None)
    cl(ws,r,3+NC,tot,fmt="#,##0",bold=True); cl(ws,r,4+NC,1,fmt="0.0%",bold=True); r+=2
    ov=sum(SRC.RES[c][bi] for c in SRC.RES_COLS for bi in (4,5,6))
    ovn=sum(SRC.RES["신규 서식"][bi] for bi in (4,5,6))
    r=legend(ws,r,[(NG,"max_px 초과 → 축소 학습"),(NEW,"신규 서식 (S17 S18)")])
    note(ws,r,4+NC,
      f"※ 학습 설정 max_px = 1,000,000 이다. 주황 구간({ov:,}행 · 전체의 {ov/n*100:.1f}%)은 학습 시 축소되어 들어간다. "
      f"특히 신규 서식은 {ovn:,}행({ovn/sum(SRC.RES['신규 서식'])*100:.0f}%)이 축소 대상이다. "
      "병합셀이 많은 표의 작은 글자가 축소로 뭉개지면 학습 효과가 떨어지므로, "
      "exp_012 결과에 따라 max_px 상향을 검토한다."); r+=3

    cl(ws,r,2,"② 정답 길이 분포 — 문자 수",bold=True,align=L)
    NC2=len(SRC.LEN_COLS)
    ws.merge_cells(start_row=r,start_column=2,end_row=r,end_column=3+NC2); r+=1
    h(ws,r,2,"길이 구간",GF)
    for i,c in enumerate(SRC.LEN_COLS,3): h(ws,r,i,c)
    h(ws,r,3+NC2,"합계",GF); r+=1
    for bi,lab in enumerate(SRC.LEN_BINS):
        cl(ws,r,2,lab,align=L); s=0
        for i,c in enumerate(SRC.LEN_COLS,3):
            v=SRC.LEN_COMPLETION[c][bi]; s+=v
            cl(ws,r,i,v,fmt="#,##0",fill=NEW if c=="신규 서식" else None)
        cl(ws,r,3+NC2,s,fmt="#,##0",bold=True); r+=1
    cl(ws,r,2,"합계",bold=True); tt=0
    for i,c in enumerate(SRC.LEN_COLS,3):
        s=sum(SRC.LEN_COMPLETION[c]); tt+=s
        cl(ws,r,i,s,fmt="#,##0",bold=True,fill=NEW if c=="신규 서식" else None)
    cl(ws,r,3+NC2,tt,fmt="#,##0",bold=True); r+=2
    note(ws,r,3+NC2,"※ max_length 8192 토큰으로 학습한다. 4,000자 이상은 신규 서식과 표에 몰려 있으며 "
                    "병합셀이 많은 표가 HTML 로 길어지기 때문이다.")

    p=f"{ROOT}/LUXIA_VLM_01_데이터셋_카탈로그.xlsx"; wb.save(p); print("✅",p)

# ══════════════ 02 학습 기록 ══════════════
def mk02():
    wb=Workbook(); ws=wb.active; ws.title="학습 기록"
    # 김덕기 주임 실험노트 컬럼 순서를 따르고, VLM 특유 항목(lora_r/max_px/가설)을 덧붙였다.
    W(ws,[3,10,13,12,9,22,9,17,10,7,7,8,7,8,8,9,9,7,7,8,9,7,15,10,10,17,26,50])
    h(ws,2,2,"exp_id",GF);      ws.merge_cells("B2:B3")
    h(ws,2,3,"desc",GF);        ws.merge_cells("C2:C3")
    h(ws,2,4,"train_date",GF);  ws.merge_cells("D2:D3")
    h(ws,2,5,"train_type",GF);  ws.merge_cells("E2:E3")
    h(ws,2,6,"model_name",GF);  ws.merge_cells("F2:F3")
    h(ws,2,7,"version",GF);     ws.merge_cells("G2:G3")
    h(ws,2,8,"base_model",GF);  ws.merge_cells("H2:H3")
    h(ws,2,9,"dataset_name",GF);ws.merge_cells("I2:I3")
    h(ws,2,10,"count",GF);      ws.merge_cells("J2:J3")
    h(ws,2,11,"training_parameter",GF); ws.merge_cells("K2:V2")
    for i,n in enumerate(["lora_r","alpha","dropout","epoch","lr","scheduler",
                          "micro_batch","accum_step","max_length","max_px","n_gpu","dtype"],11):
        h(ws,3,i,n)
    h(ws,2,23,"GPU",GF);            ws.merge_cells("W2:W3")
    h(ws,2,24,"author",GF);         ws.merge_cells("X2:X3")
    h(ws,2,25,"학습서버",GF);          ws.merge_cells("Y2:Y3")
    h(ws,2,26,"train_runtime",GF);  ws.merge_cells("Z2:Z3")
    h(ws,2,27,"checkpoint 경로",GF); ws.merge_cells("AA2:AA3")
    h(ws,2,28,"NOTE — 가설 / 결과 요약",GF); ws.merge_cells("AB2:AB3")
    ws.freeze_panes="D4"
    r=4
    for x in F.EXPS:
        eid=x[0]; m=SRC.TRAIN_META[eid]
        f=NEW if eid=="exp_012" else (HI if eid=="exp_011" else None)
        # x = (id,date,status,base,dsver,count, r,alpha,drop,ep,lr,sch,batch,accum,mlen,mpx,ngpu, gpu,runtime,ckpt,note)
        row=[eid, m[0], x[1], m[1], m[2], m[3], x[3], x[4], x[5],
             x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14],x[15],x[16], m[6],
             x[17], m[4], m[5], x[18], x[19], x[20]]
        for i,v in enumerate(row,2):
            cl(ws,r,i,v,fill=f,align=L if i in (2,3,6,8,9,23,26,27,28) else C,
               fmt="#,##0" if i==10 and isinstance(v,int) else ("0.00000" if i==15 and isinstance(v,float) else None),
               bold=(i==2 and eid in ("exp_011","exp_012")))
        r+=1
    r+=1
    r=legend(ws,r,[(NEW,"exp_012 (진행 중)"),(HI,"exp_011 (현 최고 성능)")])
    note(ws,r,28,"※ 컬럼 구성은 김덕기 주임 실험노트(Full_SFT_Evaluation.xlsx) 형식을 따랐고, "
                 "VLM 학습 특유 항목(lora_r · alpha · dropout · max_px)과 가설을 덧붙였다. "
                 "학습 설정은 exp_010 이후 유효배치 2 · lr 7e-5 로 고정해 데이터 변경 효과만 분리한다. "
                 "dataset_name 은 데이터셋 카탈로그의 2.학습 활용 데이터 시트와 매핑된다."); r+=2
    note(ws,r,28,"※ train_type 은 전 실험 LoRA 다. 베이스 모델 전체를 학습하지 않고 어댑터만 학습한 뒤 "
                 "merged_model.py 로 병합해 서빙한다. version 은 산출 모델의 배포 버전이며 exp_id 와 1:1 대응한다.")

    ws2=wb.create_sheet("요인 분리"); W(ws2,[3,12,10,12,10,10,46])
    cl(ws2,2,2,"크롭 대 페이지 반복 — 무엇이 어느 지표를 만드는가",bold=True,align=L)
    ws2.merge_cells("B2:G2")
    for i,n in enumerate(["실험","크롭","페이지 반복","CER↓","TEDS↑","해석"],2): h(ws2,3,i,n)
    r=4
    EXPL={"exp_004":"기준 — 크롭 892 · 반복 없음","exp_010":"크롭 고정, 반복만 24회 → CER 악화",
          "exp_011":"페이지 고정, 크롭 13배 → TEDS·CER 동시 개선","exp_009":"양쪽 극단 → TEDS 최고, CER 최악"}
    for e,cp,rep,cer,teds in F.FACTOR:
        f=HI if e=="exp_011" else None
        cl(ws2,r,2,e,fill=f,bold=e=="exp_011"); cl(ws2,r,3,cp,fill=f,fmt="#,##0")
        cl(ws2,r,4,rep,fill=f); cl(ws2,r,5,cer,fill=f,fmt="0.0000"); cl(ws2,r,6,teds,fill=f,fmt="0.0000")
        cl(ws2,r,7,EXPL[e],fill=f,align=L); r+=1
    r+=1
    cl(ws2,r,2,"※ 페이지 반복은 CER 을 해치고 크롭은 TEDS 를 만든다. 영수증 원본이 182장뿐이라 반복이 곧 과적합이다. "
               "exp_011 이 그 조합(크롭 증량 + 반복 억제)이다.",align=L)
    ws2.merge_cells(start_row=r,start_column=2,end_row=r,end_column=7)

    ws3=wb.create_sheet("학습 환경"); W(ws3,[3,20,52,44])
    for i,n in enumerate(["항목","내용","비고"],2): h(ws3,2,i,n)
    ENV=[("서버","52번 (192.168.250.52)","H100 80GB × 8"),
         ("베이스 모델","vlm_exp34/ml/models/Qwen3.6-35B-A3B","67GB · MoE 35B/3B활성 · vocab 248,320"),
         ("학습 프레임워크","transformers 5.15 + PEFT + accelerate FSDP","ml/fsdp_config_pure.yaml"),
         ("병합 스크립트","ml/merge_model/merged_model.py","김동영 선임 원본 무수정 복사"),
         ("서빙 이미지","luxia-chat-api-package-vllm:3.1.7","vLLM 0.20.1.dev0+g88d34c640"),
         ("서빙 인자","--max-model-len 32768 --gpu_memory_utilization 0.95 --max-num-seqs 192","GPU7 설정과 동일"),
         ("정밀도","bf16","양자화 없음 (quantization_config: None)"),
         ("LoRA 대상","250 모듈 (MoE 구조상 제한)","27B dense 는 400 모듈"),
         ("주의 — load-best","사용하지 않음","exp_009 에서 eval_loss 가 생성 품질과 반대로 움직여 checkpoint-400 이 최종으로 뽑히는 사고 발생"),
         ("검증셋","val_4dom 240행 (도메인 4종)","exp_013 부터 적용. exp_012 는 구 검증셋(영수증 90행)으로 진행 중")]
    r=3
    for x in ENV:
        for i,v in enumerate(x,2): cl(ws3,r,i,v,align=L)
        r+=1
    p=f"{ROOT}/LUXIA_VLM_02_학습기록.xlsx"; wb.save(p); print("✅",p)

# ══════════════ 03 평가 기록 ══════════════
def mk03():
    wb=Workbook(); ws=wb.active; ws.title="자체 평가 (1019 크롭)"
    W(ws,[3,15,10,10,11,10,10,46])
    h(ws,2,2,"model",GF); ws.merge_cells("B2:B3")
    h(ws,2,3,"지표",GF); ws.merge_cells("C2:G2")
    for i,n in enumerate(["CER↓","TEDS↑","완전일치↑","반복루프","표 실패"],3): h(ws,3,i,n)
    h(ws,2,8,"비고",GF); ws.merge_cells("H2:H3")
    ws.freeze_panes="C4"
    r=4
    for m,cer,teds,ex,lp,tf,nt in F.SELF:
        f=NEW if m=="exp_012" else (HI if m=="exp_011" else None)
        cl(ws,r,2,m,fill=f,align=L,bold=m in ("exp_004","exp_011","exp_012"))
        for i,v in enumerate([cer,teds,ex,lp,tf],3):
            cl(ws,r,i,v if v is not None else "-",fill=f,
               fmt=("0.0000" if i in (3,4) else ("0.0\"%\"" if i==5 else "0")) if v is not None else None)
        cl(ws,r,8,nt,fill=f,align=L); r+=1
    cl(ws,r,2,"1차 목표",fill=GOALF,align=L,bold=True)
    for i,v in enumerate([*F.GOAL,"-","-"],3): cl(ws,r,i,v,fill=GOALF,bold=True)
    cl(ws,r,8,"",fill=GOALF); r+=2
    r=legend(ws,r,[(HI,"exp_011 (현 최고)"),(NEW,"exp_012 (측정 예정)"),(GOALF,"1차 목표")])
    note(ws,r,8,"※ CER = 편집거리 / max(정답길이, 예측길이) — 0~1 유계. 2026-08-20 정의 변경 후 전 실험 재계산. "
                "구 정의는 상한이 없어 반복루프 1건이 CER 512 를 찍고 평균을 지배했다 (exp_004 구 정의 1.9313).")

    ws2=wb.create_sheet("SS 평가 (400건 KIE)")
    W(ws2,[3,30,10,9,9,9,10,9,9,9,9,10,26])
    h(ws2,2,2,"구성",GF); ws2.merge_cells("B2:B3")
    h(ws2,2,3,"파이프라인",GF); ws2.merge_cells("C2:C3")
    h(ws2,2,4,"전체 (400건)",GF); ws2.merge_cells("D2:G2")
    for i,n in enumerate(["total","general","table","행수불일치"],4): h(ws2,3,i,n)
    h(ws2,2,8,"도메인별 total",GF); ws2.merge_cells("H2:K2")
    for i,n in enumerate(["영수증","견적서","계산서","명세서"],8): h(ws2,3,i,n)
    h(ws2,2,12,"측정 배치",GF); ws2.merge_cells("L2:L3")
    h(ws2,2,13,"비고",GF); ws2.merge_cells("M2:M3")
    ws2.freeze_panes="D4"
    r=4
    for nm,pl,tot,gen,tbl,mm,rc,qu,iv,bl,bt,nt in F.SS:
        f=NEW if nm.startswith("exp_012") else (HI if "alt/high" in nm else None)
        cl(ws2,r,2,nm,fill=f,align=L,bold="alt/high" in nm); cl(ws2,r,3,pl,fill=f)
        for i,v in enumerate([tot,gen,tbl],4): cl(ws2,r,i,v if v is not None else "-",fill=f,fmt="0.000" if v is not None else None)
        cl(ws2,r,7,mm if mm is not None else "-",fill=f,fmt="0" if mm is not None else None)
        for i,v in enumerate([rc,qu,iv,bl],8): cl(ws2,r,i,v if v is not None else "-",fill=f,fmt="0.000" if v is not None else None)
        cl(ws2,r,12,F.BATCH.get(bt,"-"),fill=f); cl(ws2,r,13,nt,fill=f,align=L); r+=1
    r+=1
    r=legend(ws2,r,[(HI,"현 배포 (DocStudio alt/high)"),(NEW,"측정 예정")])
    note(ws2,r,13,"※ "+F.SS_COND)

    ws3=wb.create_sheet("실패 분석"); W(ws3,[3,9,32,7,8,8,8,8,9,15,36,34])
    cl(ws3,2,2,"1. 도메인별 오답 유형 분해   (대상: DocStudio + exp_004 출력 · SS 400건)",bold=True,align=L)
    ws3.merge_cells("B2:L2")
    h(ws3,3,2,"도메인",GF); h(ws3,3,3,"General 필드",GF)
    for i,n in enumerate(F.CATS,4): h(ws3,3,i,n)
    h(ws3,3,9,"오답 계",GF); h(ws3,3,10,"데이터로 해결",GF); h(ws3,3,11,"비고",GF)
    ws3.merge_cells("K3:L3")
    r=4
    for d,n,a,b,c_,dd,e,nt in F.DOMAIN_MIX:
        cl(ws3,r,2,d,align=L,bold=True); cl(ws3,r,3,n,fmt="#,##0")
        cl(ws3,r,4,a/100,fmt="0.0%")
        cl(ws3,r,5,b/100,fill=NG,fmt="0.0%"); cl(ws3,r,6,c_/100,fill=OK,fmt="0.0%")
        cl(ws3,r,7,dd/100,fill=NG,fmt="0.0%"); cl(ws3,r,8,e/100,fill=OK,fmt="0.0%")
        cl(ws3,r,9,(100-a)/100,fmt="0.0%"); cl(ws3,r,10,(c_+e)/100,fill=OK,fmt="0.0%",bold=True)
        cl(ws3,r,11,nt,align=L); ws3.merge_cells(start_row=r,start_column=11,end_row=r,end_column=12)
        ws3.cell(r,12).border=BD
        r+=1
    r+=1
    r=legend(ws3,r,[(OK,"학습 데이터로 줄일 수 있음"),(NG,"표기·기준 문제 — 데이터로 줄지 않음")])
    note(ws3,r,12,"※ 완전일치 외 4종이 오답이다. 과다 추출·실제오독(초록)은 학습 데이터로 줄일 수 있다. "
                  "표기차이·의미동등(주황)은 정보는 맞는데 표현이 달라 오답 처리된 것이라 데이터를 넣어도 줄지 않으며, "
                  "출력 정규화 규칙이나 채점 기준으로 다뤄야 한다."); r+=2
    cl(ws3,r,2,"2. 실패 필드별 유형   (오답 건수 상위)",bold=True,align=L)
    ws3.merge_cells(start_row=r,start_column=2,end_row=r,end_column=12); r+=1
    h(ws3,r,2,"도메인",GF); h(ws3,r,3,"실패 필드",GF); h(ws3,r,4,"오답")
    for i,n in enumerate(F.CATS[1:],5): h(ws3,r,i,n)
    h(ws3,r,9,"주 유형",GF); h(ws3,r,10,"해결",GF); h(ws3,r,11,"실체",GF); h(ws3,r,12,"조치",GF)
    r+=1
    for dom,fld,w_,a,b,c_,d_,real,act,solv in F.FIELDS:
        top=max([(F.CATS[1],a),(F.CATS[2],b),(F.CATS[3],c_),(F.CATS[4],d_)],key=lambda x:x[1])
        cl(ws3,r,2,dom); cl(ws3,r,3,fld,align=L); cl(ws3,r,4,w_,fmt="#,##0")
        for i,v in enumerate([a,b,c_,d_],5): cl(ws3,r,i,v,fmt="#,##0")
        cl(ws3,r,9,f"{top[0]} {top[1]/w_*100:.0f}%")
        cl(ws3,r,10,"데이터" if solv else "규칙",fill=OK if solv else NG,bold=True)
        cl(ws3,r,11,real,align=L); cl(ws3,r,12,act,fill=OK if solv else NG,align=L); r+=1
    cl(ws3,r,2,"계산서"); cl(ws3,r,3,"reference_numbers",align=L)
    for i in range(4,10): cl(ws3,r,i,"-")
    cl(ws3,r,10,"규칙",fill=NG,bold=True)
    cl(ws3,r,11,"리스트 타입 — 라벨-값 쌍 단위 별도 채점 필요",align=L)
    cl(ws3,r,12,"채점 로직 보완",fill=NG,align=L); r+=2
    note(ws3,r,12,"※ 유형 판정 기준 — "+F.CAT_RULE); r+=2
    note(ws3,r,12,"※ 정정 (2026-09-01) — 종전 기재 '여러 줄 주소 블록을 읽지 못함' 은 필드명 기준 추정이었다. "
                  "실제 출력을 대조한 결과 주소 자체는 정확히 읽고 같은 칸의 TEL/FAX 를 이어 붙이는 문제였다. "
                  "또한 계산서는 오답의 78%가 정보는 맞고 표현만 다른 경우여서 학습 데이터 추가로 줄지 않는다. "
                  "종전 조치란의 '계산서 전 필드 S17 S18 투입' 은 이에 따라 수정했다.")
    ws3.freeze_panes="C4"
    p=f"{ROOT}/LUXIA_VLM_03_평가기록.xlsx"; wb.save(p); print("✅",p)

if __name__=="__main__":
    mk01(); mk02(); mk03()
