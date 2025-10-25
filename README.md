# Google Colab使用教學(附帶MINST案例)
#**Google Colab 使用教程**

**「Google Colab」是什麼?**  
Google Colab是Google免費提供的Jupyter筆電環境,支援CPU、GPU和NPU處理,提供諸如 TensorFlow、pytorch、Kernal等主流深度學習框架的環境。該平台部署在雲端,不影響本地使用,因此再爛再破的電腦也依舊能夠正常使用。
Google Colab的官方網站:"https://colab.research.google.com/"

**why choices google colab**  
Google Colab為所有的開發者免費提供一定的GPU算力,每個人大約能分到一張特斯拉T4顯卡的算力,該顯卡單精度浮點的運算能力大約在2070與1080之間,同時擁有16G顯存,如果自己擁有更好的顯卡(ex:4080的顯卡)那用自己的會更好。如果自己電腦為3060的,雖然單精度浮點的運算能力比T4強,但出於顯存考量以及自己筆電經常外帶的需求,會建議使用該平台會更好。 由於是免費提供的,因此該算力也有限制,即每週最多使用三十小時左右(大概,官方也沒有公佈限額,這是動態資源),同時單次運行不能超過12小時,同時若使用用戶過多的情況下不一定能使用上。
Colab Pro 訂閱者的使用量仍會受到限制,但相比非訂閱者可享有的限額要多出約一倍。Colab Pro+ 訂閱者還可獲得更高的穩定性。

**what is Google Driver?**  
Google Driver是Google推出的線上儲存服務,就是所謂的雲端硬碟,目前有付費和免費兩種模式,免費用戶可享有15G的空間,付費用戶根據方案最多可享有20TB的空間。 Google Driver: "https://drive.google.com/drive/"

**why used Google Colab ?**  
如上文所說,Google Colab是谷歌免費提供的Jupyter筆電環境,在每次關閉該環境,伺服器會自動將之前的所有操作進行清除,如果不使用Google Driver,則每次都需要上傳數據集和代碼,非常浪費時間,因此使用雲端 平台,和Colab進行鏈接操作,在使用Colab的時候可以調整網盤的數據。

**正式教學**

**create notebook**

首先進入Google Driver: https://drive.google.com/drive/
<img src="https://github.com/kity2233466/KAKA/blob/main/DRIVE_0.jpg">

點選左上角的「新建」，然後點選「更多」
<img src="https://github.com/kity2233466/KAKA/blob/main/2_0.jpg">

通常不一定會看到Google Colaboratory,沒有的話點擊“關聯更多應用程式”,搜尋“Colab”,安裝第一個即可
<img src="https://github.com/kity2233466/KAKA/blob/main/3_0.jpg">

下載後，點擊進入Colab
<img src="https://github.com/kity2233466/KAKA/blob/main/4_0.jpg">

<img src="https://github.com/kity2233466/KAKA/blob/main/5_0.jpg">

若直接點選Colab的網址則為該頁面
<img src="https://github.com/kity2233466/KAKA/blob/main/%E9%80%B2%E5%85%A5%E7%95%AB%E9%9D%A2.jpg">

這時你只需要點選左上角的檔案，點擊「在雲端硬碟中建立新筆記本」就可以進入相同的頁面
<img src="https://github.com/kity2233466/KAKA/blob/main/%E5%BB%BA%E7%AB%8B%E6%96%B0%E7%AD%86%E8%A8%98%E6%9C%AC.jpg">

<img src="https://github.com/kity2233466/KAKA/blob/main/5_0.jpg">

左邊有五個選項,分別為目錄、尋找和替換、變數、Secret(秘鑰)、檔案

之後點選上面執行階段,然後點選變更執行階段類型,在其中硬體加速器部分選擇T4 GPU後，儲存設定，
Colab便會配置一個帶有GPU的機器,此時筆記本就創建完成了。
<img src="https://github.com/kity2233466/KAKA/blob/main/6_0.jpg">

<img src="https://github.com/kity2233466/KAKA/blob/main/7_0.jpg">

###雲端硬碟掛載由於我們的資料集以及程式碼檔案都放在了Google雲端碟上,因此還需要對Google 雲端硬碟進行掛載在新建立的筆記本中輸入以下程式碼

<img src="https://github.com/kity2233466/KAKA/blob/main/%E6%8E%9B%E8%BC%89%E7%A8%8B%E5%BC%8F%E7%A2%BC.jpg">

運行後便可以獲得該視窗

<img src="https://github.com/kity2233466/KAKA/blob/main/8_0.jpg">

之後在一系列的視窗中進行登入Google帳號,同時授權對雲端磁碟檔案的讀取與修改,完成後便可實現掛載

<img src="https://github.com/kity2233466/KAKA/blob/main/9_0.jpg">

<img src="https://github.com/kity2233466/KAKA/blob/main/10_0.jpg">

可以看到的是,我們谷歌硬碟裡面的資料已經放在./gdrive/MyDrive 這個目錄裡面,我們再去呼叫
的時候就會十分方便。

**use 命令列**  

在notebook環境下,你只需要在每一行程式碼前面多加一個「!」(注意是英文的感嘆號),便可以
像Linux系統裡的終端指令操作那樣進行指令的輸入

EX:使用Is指令,便可以得到目前目錄下的路徑

<img src="https://github.com/kity2233466/KAKA/blob/main/11.jpg">

**以MINST手寫數字資料集作為範例進行訓練**

**CPU版**
<img src="https://github.com/kity2233466/KAKA/blob/main/1-1.jpg">

導入相對應的庫

<img src="https://github.com/kity2233466/KAKA/blob/main/1-2.jpg">

神經網路的建立

<img src="https://github.com/kity2233466/KAKA/blob/main/1-3.jpg">

資料集下載(如果沒有的話會自動下載；有的話會自動跳過並讀取到相對應的資料)

<img src="https://github.com/kity2233466/KAKA/blob/main/1-4.jpg">

載入資料集(每十張圖片為一批,並隨機打亂)

<img src="https://github.com/kity2233466/KAKA/blob/main/16.jpg">

網路實例化

<img src="https://github.com/kity2233466/KAKA/blob/main/1-5.jpg">

參數優化、學習率與訓練輪次設定

<img src="https://github.com/kity2233466/KAKA/blob/main/1-6.jpg">

開始訓練

<img src="https://github.com/kity2233466/KAKA/blob/main/1-7.jpg">

模型驗證

<img src="https://github.com/kity2233466/KAKA/blob/main/1-8.jpg">

訓練結果如圖
<img src="">


具體的程式碼請參考./notebook/CPU版該文件

**GPU版**
如果你已經參考CPU版的程式碼使其成功跑起來的話,你會留意到一件事:為什麼訓練這麼慢? 這時因為我們使用的是CPU去跑,接下來我們就用GPU去跑

首先將更改運行類型,依序點擊程式碼執行程序-更改運行時類型便可以得到以下窗口
<img src="https://github.com/kity2233466/KAKA/blob/main/7_0.jpg">

切換運行類型後倒入庫的同時讀取設備id
<img src="https://github.com/kity2233466/KAKA/blob/main/gpu%E5%B0%8E%E5%85%A5%E5%BA%AB.jpg">

神經網路建立

<img src="https://github.com/kity2233466/KAKA/blob/main/14.jpg">

資料集下載

<img src="https://github.com/kity2233466/KAKA/blob/main/15.jpg">

載入資料集,設定每十張照片為一批,並隨機打亂

<img src="https://github.com/kity2233466/KAKA/blob/main/16.jpg">

網路實例化

<img src="https://github.com/kity2233466/KAKA/blob/main/17.jpg">

優化器、學習率、輪次設定

<img src="">
開始訓練

<img src="">

模型測試

<img src="">

訓練結果如圖
<img src="">
