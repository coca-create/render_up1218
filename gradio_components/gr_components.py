import gradio as gr
from tab1 import tab1_func as t1
from tab2 import tab2_func as t2
from tab3 import tab3_func as t3
from tab4 import tab4_func as t4
from tab5 import tab5_func as t5
#from tab6 import tab6_func_stock as t6
from tab7 import tab7_func as t7
from tab8 import tab8_func as t8
from Jsons import jsonplussrt as rev_j
import pandas as pd



def gr_components():

    with gr.Blocks() as UI:
        gr.Markdown(
            """
            <h1 style="color:'darkblue'; font-family :'Arial', sans-serif;font-size:36px;"> PeriOz　- Web & Trial - </h1>
            <p style="color:gray; letter-spacing:0.05em;">OpenAIのfaster-whisperを使っています。字幕の区切りを必ずピリオドにできるのがこのアプリの特徴です。それにより翻訳精度が保たれます。<br>一方、この方式を使うデメリットは一文が長いこと。日本語字幕作成後は「Subtitle Edit」等の自動分割を利用すると読みやすくなります。</p>
            """)
        ### Gradio-Tab7 ###

           
        with gr.Tab("配布字幕の再編"):
               
            gr.Markdown('> "翻訳お手伝い②"とは異なり、ピリオドを基準に字幕ファイルを再編、翻訳に使います。ピリオド区切りのファイルは字幕の区切りが長くなるため、Subtitle Editなどのアプリで「長い文の自動分割機能」を使うと読みやすくなります。例えば、HealingAlSのvimeo動画に付属する字幕をピリオド区切りに再編し、さらに翻訳した字幕ファイルを作ります')
            with gr.Row(equal_height=True):
                vtt_input = gr.File(label="vtt/srtファイルをアップロードしてください。")  # input用のoriginal vtt,srt ",file_types=['srt','vtt']
                with gr.Column():
                    vtt_output_2 = gr.File(label="ピリオド区切りの英語字幕ファイルとワードファイルです。",file_count="multiple")  # 分割・結合処理後のvtt,srtファイル
                    vtt_translated_file = gr.File(label="ピリオド区切りの英文から作った日本語字幕ファイルです。")  # 翻訳されたvtt,srtファイルの出力
            with gr.Row():
                t7_clear_button = gr.Button("クリア")  
                t7_translate_button = gr.Button("日本語vtt,srtの作成",variant='primary')
            with gr.Row():
                vtt_output_1 = gr.HTML()  # 分割・結合処理後のHTML表示
                vtt_translated_content = gr.TextArea(label="翻訳された字幕情報を貼り付けてください。")  # 翻訳処理後の内容を貼り付け。 
                dummy_file=gr.File(visible=False)       

            with gr.Column():
                with gr.Accordion(label="英語dataframe" ,open=False):
                    t7_dataframe=gr.HTML()        

        with gr.Tab("翻訳お手伝い"):
        
            gr.Markdown(">ピリオドに基づく再編は含まれません。主にtxtファイル、ピリオド再編が不要なsrt,vttファイルが対象です。srt,vtt,txtのいずれかの英文ファイルをアップロードすると内容が表示されます。次にGoogle翻訳で得た翻訳をテキストエリアに入力します。「翻訳ファイルを作成」ボタンを押して、入力ファイルと同形式のファイルに保存します。ファイル名に _ja が付加されます。")
            with gr.Row(equal_height=True):
                file_input = gr.File(label="Upload file", file_count="single")# ,file_types=['.txt','.srt','.vtt']
                with gr.Column():
                    t4_excel_path=gr.File(label="Excel or Word file for Google translate",type="filepath",file_count="multiple")
                    output_file = gr.File(label="Translated file" ,type='filepath',file_count="multiple")

            with gr.Row():
                t4_clear_button=gr.Button("クリア")
                translate_button = gr.Button("翻訳ファイル作成", variant='primary')
            
            with gr.Row():
                file_content = gr.HTML(label="File content")
                translated_text = gr.TextArea(label="Translated text")
            with gr.Column():
                with gr.Accordion(label="英語dataframe" ,open=False):

                    t4_dataframe=gr.HTML()

        with gr.Tab("whisperファイルの復活"):
            gr.Markdown('> whisperがピリオドを打たなくなったSRTファイルの内容にピリオドを追加して、ピリオド区切りのファイルを生成します。')            
            with gr.Row(equal_height=True):
                with gr.Column():
                    jsonfile=gr.File(label="jsonファイルをアップロードしてください。",type='filepath',file_count="multiple")
                    srtfile=gr.File(label="srtファイルをアップロードしてください。",type='filepath',file_count="multiple")
                revj=gr.File(label="修復後のファイルです。",type='filepath',file_count="multiple")
            with gr.Row():
                json_execute=gr.Button("ファイルの修復",variant='primary')
                json_clear=gr.Button("クリア")
         
        ### Gradio-Tab5 ###
        with gr.Tab("Word/Excel↔SRT/VTT/TXT"):
            gr.Markdown("> 日本語のword,Excelファイルをsrt/vtt/txt形式に戻すためのプログラムです。wordファイルは末尾が[_srt.xlsx][_vtt.xlsx][_txtnr.docx],[_txtr.docx]、あるいは[_srt (1).xlsx]のように（1）の付加された日本語ファイルが対象です。複数のファイルを一度に扱えますが、アップロードは1回で行う必要があります。")  
            with gr.Column():
                with gr.Row():
                    to_srttxt_input = gr.File(label="Upload docx/xlsx for srt/txt", file_count="multiple", type='filepath')#,file_types=["docx","xlsx"]
                    to_srttxt_output = gr.File(label="Converted srt/vtt/txt")
                with gr.Row():
                    to_srttxt_button = gr.Button("DOCX/XLSX　→　SRT/TXT", variant='primary')
                    to_srttxt_clear_button = gr.Button("クリア")

            gr.Markdown("> 翻訳準備として、英語のsrt,vttあるいはtxtファイルをword,excel形式に変換するためのプログラムです。srt,vttまたはtxtファイルは末尾が[.srt][.vtt][_NR.txt][_R.txt]のファイルのみ入力できます。複数のファイルを一度に扱えますが、アップロードは1回で行う必要があります。")
            with gr.Row():
                various_file_input = gr.File(file_count='multiple', label="Upload srt/vtt/txt for docx/xlsx")#,file_types=['srt','vtt','txt']
                output_doc_files = gr.File(file_count='multiple', label="Converted doc/xlsx")
                
            with gr.Row():
                submit_transform_button = gr.Button("SRT/VTT/TXT　→　DOCX/XLSX",variant='primary')
                clear_transform_button = gr.Button("クリア")
                 
        
        

        #tab8
            
        with gr.Tab("日本語srt/vttの句点分割"):
            gr.Markdown("> 日本語字幕ファイルを句点分割します。セグメントが長すぎる字幕に使ってみましょう。これでも長い場合はSubtitle Editをご利用ください。") 
            with gr.Row():
                input_files = gr.File(label="Upload SRT/VTT Files", file_count="multiple")#,file_types=['srt','vtt']
                output_files = gr.File(label="Download SRT/VTT Files")
            with gr.Row():
                process_btn = gr.Button("句点分割",variant="primary")
                clear_btn = gr.Button("Clear")

            process_btn.click(t8.process_and_display, inputs=[input_files], outputs=[output_files])
            clear_btn.click(t8.clear_files, outputs=[input_files, output_files])
        ### Gradio-Tab3 ###
        
        with gr.Tab("SRT/VTT→Excel(2言語)"):
            gr.Markdown("> 英語と日本語を並べて読むためのツールです。文字起こしの際に作成できるexcelファイルと同じです。2つのsrtファイルはタイムスタンプが一致している必要があります。")   
            lang_for_xls_choice = gr.Radio(
                choices=["English and Japanese", "only English", "only Japanese"],
                label="どんなExcelファイルを作りますか？",
                interactive=True,
                value="English and Japanese"
            )
            with gr.Row(equal_height=True):
                english_file = gr.File(label="英語のSRT,VTTファイルをアップロード", visible=True)#,file_types=['srt','vtt']
                japanese_file = gr.File(label="日本語のSRT,VTTファイルをアップロード", visible=True)#,file_types=['srt','vtt']
                
            with gr.Row():        
                submit_button = gr.Button("Excelファイル作成",variant='primary')
                clear_button = gr.Button("クリア")   
            excel_output = gr.File(label="Excelファイルをダウンロード。")       
            dataframe_output = gr.HTML()
        
            
            # クリアボタンをクリックした時にファイルをクリア
            clear_button.click(
                fn=t3.clear_all_files,
                outputs=[english_file, japanese_file, dataframe_output, excel_output]
            )

            # ラジオボタンの選択変更でファイルのクリアとコンポーネントの表示を更新
            lang_for_xls_choice.change(
                fn=t3.update_visibility_and_clear,
                inputs=[lang_for_xls_choice],
                outputs=[english_file, japanese_file, dataframe_output, excel_output]
            )
            
        # ファイルを処理してデータフレームとExcelファイルを生成する
        def process_files(english_file, japanese_file, choice):
            if english_file is None and japanese_file is None:
                return pd.DataFrame({'1': [''], '2': [''], '3': ['']}), None
            
            
            if choice == "only English":
                excel_path, df = t3.create_excel_from_srt(english_path=english_file,tail="")
            elif choice == "only Japanese":
                excel_path, df = t3.create_excel_from_srt(japanese_path=japanese_file,tail="")
            else:  # "English and Japanese"
                excel_path, df = t3.create_excel_from_srt(english_path=english_file, japanese_path=japanese_file,tail="")
            
            df=t1.dataframe_to_html_table(df)
            df=f"""
                <div class="my-table-container">
                    {df}
                </div>
            """                        
            
            
            return df, excel_path
        
        #tab9 License
        with gr.Tab("LICENSES"):
            gr.Markdown("> LICENSE条項です。お読みの上、ご利用ください。")
            # mdファイルを読み込む
            # mdファイルを読み込んでHTMLに変換する
            import markdown2
            def load_md_to_html():
                with open("LICENSES.md", "r", encoding="utf-8") as file:
                    content = file.read()
                # markdown2のtables拡張機能を有効にしてHTMLに変換
                html_content = markdown2.markdown(content, extras=["tables"])
                return html_content

            gr.HTML(load_md_to_html)




        ##クリアボタン追加分をまとめる。
        '''def t1_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,"","","",[],[],"","","","","",empty_html_table
        def t2_clear():
            return "","","",[],pd.DataFrame({'1': [''], '2': [''],'3': ['']})'''
        def t4_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,"","",[],[],empty_html_table

        '''def t6_clear():
            return None,None,[]'''
        
        def t7_clear():
            empty_html_table = pd.DataFrame({'1': [''], '2': [''], '3': ['']}).to_html(index=False)
            return None,None,None,None,None,None,empty_html_table

        def param1_change_clear():
            return None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None

        ### Tab1 イベントリスナー ###
        '''param1.change(fn=param1_change_clear,
                      inputs=[],
                      outputs=[result_srt_content,result_txt_nr_content,result_txt_r_content
                               ,main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df,
                               translate_srt,translate_nr_txt,translate_r_txt,download_translated_files,button2_df])
        exec_btn.click(
            fn=t1.transcribe,
            inputs=[param1, param2, param3, param4, param5, param6,param0],
            outputs=[result_srt_content,result_txt_nr_content, result_txt_r_content, main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df])
        
        t1_clear_Button.click(
            fn=t1_clear,inputs=[],outputs=[param1,result_srt_content,result_txt_nr_content,result_txt_r_content,main_files_path,doc_download_path,html_srt,html_nr_txt,html_r_txt,filename_output,dummy,gr_components_df]
        )
        ### Tab2 イベントリスナー ###
        
        extension_choices.change(fn=update_visibility, 
                                inputs=extension_choices,
                                outputs=[translate_srt, translate_nr_txt, translate_r_txt,html_srt,html_nr_txt,html_r_txt])
        
        generate_files_button.click(
            fn=t2.create_translate_files,
            inputs=[filename_output, 
                    translate_srt,
                    translate_nr_txt,
                    translate_r_txt, 
                    extension_choices,
                    dummy],
            outputs=[download_translated_files,button2_df]) 
        t2_clear_button.click(fn=t2_clear,inputs=[],outputs=[translate_srt,translate_nr_txt,translate_r_txt,download_translated_files,button2_df])'''
        ### Tab3 イベントリスナー　###
        submit_button.click(
                fn=process_files,
                inputs=[english_file, japanese_file, lang_for_xls_choice],
                outputs=[dataframe_output, excel_output]
            )

        ### Tab4 イベントリスナー ###
        file_input.change(fn=t4.display_file_content, inputs=file_input, outputs=[file_content, t4_excel_path,t4_dataframe])

        translate_button.click(fn=t4.translate, inputs=[file_input, translated_text], outputs=output_file)

        t4_clear_button.click(fn=t4_clear,inputs=[],outputs=[file_input,file_content,translated_text,output_file,t4_excel_path,t4_dataframe])
        ### Tab5 イベントリスナー ###
        to_srttxt_button.click(
        fn=t5.convert_docx_to_srttxt,
        inputs=to_srttxt_input,
        outputs=to_srttxt_output
        )
        
        to_srttxt_clear_button.click(
            fn=t5.clear_inputs,
            inputs=[],
            outputs=[to_srttxt_input, to_srttxt_output]
        )

        submit_transform_button.click(t5.process_doc_files, inputs=various_file_input, outputs=output_doc_files)
        clear_transform_button.click(t5.clear_both, inputs=None, outputs=[various_file_input, output_doc_files])   
        ### Tab6 イベントリスナー ###
        '''generate_voice.click(
            fn=t6.tts,
            inputs=[input_audio,voice_select],
            outputs=[output_audio,download_audio] )
        t6_clear_button.click(
            fn=t6_clear,
            inputs=[],
            outputs=[input_audio,output_audio,download_audio]
        )'''
        ### Tab7 イベントリスナー ###
        vtt_input.upload(
        fn=t7.process_file, inputs=[vtt_input], outputs=[vtt_output_1, vtt_output_2,dummy_file,t7_dataframe])
        t7_translate_button.click(
            fn=t7.vtt_translate,
            inputs=[vtt_input, vtt_translated_content,dummy_file],
            outputs=[vtt_translated_file]
        )
        t7_clear_button.click(
            fn=t7_clear,
            inputs=[],
            outputs=[vtt_input,vtt_translated_content,vtt_translated_file,vtt_output_1,vtt_output_2,dummy_file,t7_dataframe]
        )
        def fn_j_clear():
            return [],[],[]
        
        json_clear.click(
            fn=fn_j_clear,
            inputs=[],
            outputs=[jsonfile,srtfile,revj]
        )
        json_execute.click(
            fn=rev_j.repair,
            inputs=[jsonfile,srtfile],
            outputs=[revj]
        )
    
        return UI