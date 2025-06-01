import streamlit as st
from rag_backend import ask_rag, get_system_info, search_documents
from config import MODEL_NAME, DATA_DIR
from pathlib import Path
import json

# Page configuration
st.set_page_config(
    page_title="Chatbot Asisten Kontraktor", 
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #ff6b35;
    background-color: #f8f9fa;
}
.source-box {
    background-color: #e9ecef;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem 0;
    font-size: 0.9rem;
}
.stats-card {
    background-color: #f1f3f4;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("Chatbot Asisten Kontraktor")
st.caption("Asisten AI untuk industri konstruksi - Konsultasi SOP, regulasi, estimasi biaya, dan manajemen proyek")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi Sistem")
    
    # System info
    system_info = get_system_info()
    
    st.markdown("### 📊 Status Sistem")
    if system_info["status"] == "loaded":
        st.success("✅ Sistem Siap")
        st.info(f"📄 Dokumen: {system_info['index_size']}")
        st.info(f"🤖 Model: {MODEL_NAME}")
        st.info(f"📂 Sumber Data: {DATA_DIR}")
    else:
        st.error("Error Sistem")
        st.error(system_info.get("error", "Error tidak diketahui"))
    
    st.markdown("### Opsi Lanjutan")
    
    # Search mode
    search_mode = st.selectbox(
        "Mode Pencarian",
        ["Tanya Jawab dengan AI", "Pencarian Dokumen Saja"],
        help="Pilih antara Q&A dengan AI atau pencarian dokumen sederhana"
    )
    

    k_docs = st.slider(
        "Jumlah Dokumen yang Diambil",
        min_value=1,
        max_value=16,
        value=16,
        help="Jumlah dokumen relevan untuk dijadikan referensi"
    )
    
    st.markdown("---")
    st.markdown("### Shortcut")
    
    if st.button("Refresh Sistem"):
        st.cache_data.clear()
        st.rerun()
    
    if st.button("Statistik Dokumen"):
        st.session_state.show_stats = True

#main
col1, col2 = st.columns([2, 1])

with col1:
    #init chat history
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # query input
    query = st.text_input(
        "Masukkan pertanyaan Anda:",
        placeholder="Contoh: Bagaimana prosedur K3 untuk pekerjaan di ketinggian?",
        key="query_input"
    )
    
    # search
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    
    with col_btn1:
        search_button = st.button("Search", type="primary", use_container_width=True)
    
    with col_btn2:
        clear_button = st.button("Hapus", use_container_width=True)
    
    if clear_button:
        st.session_state.history = []
        st.rerun()
    
    # process
    if search_button and query:
        with st.spinner("🔍 Mencari di database pengetahuan..."):
            if search_mode == "Tanya Jawab dengan AI":
                answer, source_snippets, detailed_sources = ask_rag(query)
                
                st.session_state.history.append({
                    "type": "qa",
                    "query": query,
                    "answer": answer,
                    "sources": source_snippets,
                    "detailed_sources": detailed_sources
                })
            else:
                #cari dokumen tok
                search_results = search_documents(query, k_docs)
                
                st.session_state.history.append({
                    "type": "search",
                    "query": query,
                    "results": search_results
                })

    #show chat history
    if st.session_state.history:
        st.markdown("## Riwayat Percakapan")
        
        for idx, item in enumerate(reversed(st.session_state.history)):
            with st.container():
                st.markdown(f"### Pertanyaan {len(st.session_state.history)-idx}")
                st.markdown(f"**Pertanyaan:** {item['query']}")
                
                if item["type"] == "qa":
                    st.markdown(f"**Jawaban:**")
                    st.markdown(item["answer"])
                    
                    if item["sources"]:
                        with st.expander(f"Sumber ({len(item['sources'])})"):
                            for i, source in enumerate(item["sources"], 1):
                                st.markdown(f"**Sumber {i}:**")
                                st.markdown(source)
                                st.markdown("---")
                
                elif item["type"] == "search":
                    st.markdown(f"**Hasil Pencarian:** {len(item['results'])} dokumen ditemukan")
                    
                    with st.expander(f"📄 Hasil Dokumen ({len(item['results'])})"):
                        for i, result in enumerate(item["results"], 1):
                            st.markdown(f"**Dokumen {i}** (Skor: {result['similarity_score']:.3f})")
                            st.markdown(f"**File:** {result['metadata'].get('filename', 'Tidak diketahui')}")
                            st.markdown(f"**Preview Konten:** {result['content'][:200]}...")
                            st.markdown("---")
                
                st.divider()

with col2:
    st.markdown("## Statistik Sistem")
    
    # Show only if request on sidebar
    if st.session_state.get("show_stats", False):
        try:
            from document_processor import DocumentProcessor
            processor = DocumentProcessor()
            stats = processor.get_file_stats(DATA_DIR)
            
            st.markdown("### Statistik File")
            
            # metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric("Total File", stats['total_files'])
            with col_metric2:
                st.metric("File Didukung", stats['supported_files'])
            
            # File types
            if stats['file_types']:
                st.markdown("### Jenis File")
                for file_type, count in stats['file_types'].items():
                    st.write(f"**{file_type}**: {count} file")
            
            if stats['directories']:
                st.markdown("### Direktori")
                for directory in stats['directories']:
                    dir_name = directory if directory != '.' else 'Root'
                    st.write(f"- {dir_name}")
                    
        except Exception as e:
            st.error(f"Error memuat statistik: {str(e)}")
        
        if st.button("Sembunyikan Statistik"):
            st.session_state.show_stats = False
            st.rerun()
    
    # Recent 
    if st.session_state.history:
        st.markdown("### Pertanyaan Terbaru")
        for item in st.session_state.history[-3:]:
            st.markdown(f"- {item['query'][:50]}..." if len(item['query']) > 50 else f"- {item['query']}")
    
    st.markdown("### Tips Penggunaan")
    st.markdown("""
    - Gunakan istilah konstruksi yang spesifik
    - Tanyakan tentang SOP, K3, estimasi biaya
    - Sebutkan jenis pekerjaan (sipil, MEP, finishing)
    - Gunakan mode "Tanya Jawab" untuk penjelasan detail
    - Gunakan "Pencarian Dokumen" untuk referensi cepat
    """)

st.markdown("---")
st.markdown("🏗️ Chatbot Asisten Kontraktor - Dibangun dengan Punk menggunakan LangChain, FAISS, dan Streamlit")