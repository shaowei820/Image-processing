# Image-processing
## breakdown ![breakdown](https://github.com/user-attachments/assets/4e41be5d-2c28-4e5f-903c-014517b82a8f)

## API
1. **前處理**  
   - 灰階轉換與邊緣檢測
2. **紋理分析**  
   - 計算 LBP 紋理特徵與直方圖
3. **路徑搜尋**  
   - 利用 BFS 演算法進行道路區域分割
4. **上色**  
   - 將道路區域標記為紅色
5. **參數**
   - 各式參數


#### 1.前處理:對影像進行灰階轉換和 Sobel 邊緣檢測。
preprocess(image)

* 輸入:
image: 彩色影像 (numpy.ndarray, BGR 格式)
* 輸出:
gray: 灰階影像 (numpy.ndarray)  edges: Sobel 邊緣檢測結果 (numpy.ndarray)   

#### 2.紋理分析 計算局部二值模式（LBP）紋理，並生成正規化的直方圖
lbp_image, lbp_hist = api.lbp_analysis(gray)

* 輸入:
gray: 灰階影像 (numpy.ndarray)
* 輸出:
lbp_image: LBP 紋理影像 (numpy.ndarray)  lbp_hist: LBP 特徵的正規化直方圖 (numpy.ndarray)

#### 3.路徑搜尋 利用 BFS 搜索演算法對道路區域進行分割。

bfs_mask = api.path_search(gray, seed_point)
* 輸入:
gray: 灰階影像 (numpy.ndarray)  seed_point: BFS 搜索的起始點座標，格式為 (x, y)
* 輸出:
bfs_mask: 分割出的道路區域 (numpy.ndarray)
#### 4.上色 將細化後的道路區域以紅色標記，只保留較大的區域。
* 輸入:
image: 彩色影像 (numpy.ndarray, BGR 格式)  road_mask: 經細化的道路遮罩 (numpy.ndarray)
* 輸出:
colored_image: 標記紅色道路的影像 (numpy.ndarray)
#### 5.參數
* 1.sobel_ksize 前處理 Sobel 邊緣檢測的核心大小（如 3x3）。
* 2.lbp_threshold 紋理分析 用於紋理分析的 LBP 閾值。
* 3.bfs_threshold 路徑搜尋 BFS 搜索的亮度閾值，控制道路區域的判斷。

## DEMO 
![image](https://github.com/user-attachments/assets/25b77875-68d7-4820-bb84-0cfaa3123fcc)

