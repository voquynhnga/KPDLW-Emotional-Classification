const productUrl1Input = document.getElementById('productUrl1');
const productUrl2Input = document.getElementById('productUrl2');
const productUrl1Label = document.getElementById('productUrl1Label');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingDiv = document.getElementById('loading');
const errorSection = document.getElementById('errorSection');

const resultsContainer = document.getElementById('resultsContainer');
const resultsSection1 = document.getElementById('resultsSection1');
const productTitle1 = document.getElementById('productTitle1');
const commentsSection1 = document.getElementById('commentsSection1');
const recommendationResult1 = document.getElementById('recommendationResult1');
const recommendationReason1 = document.getElementById('recommendationReason1');

const resultsSection2 = document.getElementById('resultsSection2');
const productTitle2 = document.getElementById('productTitle2');
const commentsSection2 = document.getElementById('commentsSection2');
const recommendationResult2 = document.getElementById('recommendationResult2');
const recommendationReason2 = document.getElementById('recommendationReason2');

const comparisonSection = document.getElementById('comparisonSection');
const comparisonResultText = document.getElementById('comparisonResultText');
const comparisonProd1Stats = document.getElementById('comparisonProd1Stats');
const comparisonProd2Stats = document.getElementById('comparisonProd2Stats');

const product1Group = document.getElementById('product1Group');
const product2Group = document.getElementById('product2Group');
const singleModeBtn = document.getElementById('singleModeBtn');
const compareModeBtn = document.getElementById('compareModeBtn');

let sentimentChart1 = null;
let sentimentChart2 = null;
let currentMode = 'single'; // 'single' or 'compare'

function setMode(mode) {
    currentMode = mode;
    clearAllResults();
    productUrl1Input.value = '';
    productUrl2Input.value = '';


    if (mode === 'single') {
        singleModeBtn.classList.add('active');
        compareModeBtn.classList.remove('active');
        product2Group.style.display = 'none';
        productUrl1Label.textContent = '🔗 Link sản phẩm (Điện máy xanh hoặc Tiki):';
        analyzeBtn.textContent = 'Phân tích';
        productTitle1.textContent = 'Kết quả Phân tích';
    } else { // compare mode
        singleModeBtn.classList.remove('active');
        compareModeBtn.classList.add('active');
        product2Group.style.display = 'block';
        productUrl1Label.textContent = '🔗 Link sản phẩm 1 (Điện máy xanh hoặc Tiki):';
        analyzeBtn.textContent = 'So sánh sản phẩm';
        productTitle1.textContent = 'Kết quả cho Sản phẩm 1';
    }
}
// Initialize mode
setMode('single');


productUrl1Input.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        startAnalysis();
    }
});
productUrl2Input.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        startAnalysis();
    }
});


async function startAnalysis() {
    const url1 = productUrl1Input.value.trim();
    const url2 = currentMode === 'compare' ? productUrl2Input.value.trim() : '';

    clearAllResults(); // Clear previous results before new analysis

    if (!url1) {
        showError('Vui lòng nhập link sản phẩm thứ nhất.');
        return;
    }
    if (!isValidUrl(url1)) {
        showError('Link sản phẩm 1 không hợp lệ. Vui lòng kiểm tra lại.');
        return;
    }
    if (currentMode === 'compare' && !url2) {
        showError('Vui lòng nhập link sản phẩm thứ hai để so sánh.');
        return;
    }
    if (currentMode === 'compare' && url2 && !isValidUrl(url2)) {
        showError('Link sản phẩm 2 không hợp lệ. Vui lòng kiểm tra lại.');
        return;
    }
    if (currentMode === 'compare' && url1 === url2 && url1 !== '') {
        showError('Vui lòng nhập hai link sản phẩm khác nhau để so sánh.');
        return;
    }


    loadingDiv.style.display = 'block';
    analyzeBtn.disabled = true;
    resultsContainer.style.display = 'none'; 


    try {
        const data1 = await fetchAnalysis(url1, 1);

        if (!data1) { 
             throw new Error("Không thể phân tích sản phẩm 1."); 
        }

        resultsContainer.style.display = 'block';
        displaySingleProductResults(data1, 1);
        resultsSection1.style.display = 'block';


        if (currentMode === 'compare' && url2) {
            const data2 = await fetchAnalysis(url2, 2);
            if (data2) {
                displaySingleProductResults(data2, 2);
                resultsSection2.style.display = 'block';
                performComparison(data1, data2);
                comparisonSection.style.display = 'block';
            } else {
                comparisonSection.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Overall Analysis Error:', error);
         if (!errorSection.textContent.includes(error.message)) { 
             showError(errorSection.textContent ? errorSection.textContent + "<br>" + error.message : error.message);
         }
    } finally {
        loadingDiv.style.display = 'none';
        analyzeBtn.disabled = false;
    }
}

function isValidUrl(string) {
    if (!string) return false;
    try {
        new URL(string); // Basic check for URL format
        // More specific check for allowed domains
        return (string.includes('dienmayxanh.com') || string.includes('tiki.vn'));
    } catch (_) {
        return false;
    }
}


async function fetchAnalysis(url, productNumber) {
    if (!url) return null;

    try {
        const response = await fetch('/analyze-product', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url }),
            signal: AbortSignal.timeout(1200000)
        });

        if (!response.ok) {
            let errorMsg = `Sản phẩm ${productNumber} - Lỗi ${response.status}: `;
            try {
                const errorData = await response.json();
                errorMsg += errorData.error || 'Không thể xử lý yêu cầu.';
            } catch {
                errorMsg += 'Phản hồi không hợp lệ từ máy chủ.';
            }
            throw new Error(errorMsg);
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(`Sản phẩm ${productNumber}: ${data.error}`);
        }
        return data;

    } catch (error) {
        console.error(`Analysis Error for Product ${productNumber}:`, error);
        let displayError = error.message;
        // Append to existing errors if any, otherwise set it.
        const currentErrorText = errorSection.innerHTML; // Use innerHTML to preserve <br>
        showError(currentErrorText ? currentErrorText + "<br>" + displayError : displayError);
        return null;
    }
}


function displaySingleProductResults(data, productNumber) {
    let chartCanvasId, currentCommentsSection, currentRecResult, currentRecReason, currentProductTitleElem, chartInstanceVarSetter;

    if (productNumber === 1) {
        chartCanvasId = 'sentimentChart1';
        currentCommentsSection = commentsSection1;
        currentRecResult = recommendationResult1;
        currentRecReason = recommendationReason1;
        currentProductTitleElem = productTitle1;
        chartInstanceVarSetter = (instance) => { sentimentChart1 = instance; };
        currentProductTitleElem.textContent = currentMode === 'compare' ? `Kết quả cho Sản phẩm 1` : `Kết quả Phân tích (${truncateUrl(data.product_url)})`;

    } else {
        chartCanvasId = 'sentimentChart2';
        currentCommentsSection = commentsSection2;
        currentRecResult = recommendationResult2;
        currentRecReason = recommendationReason2;
        currentProductTitleElem = productTitle2;
        chartInstanceVarSetter = (instance) => { sentimentChart2 = instance; };
        currentProductTitleElem.textContent = `Kết quả cho Sản phẩm 2`;
    }

    const existingChartInstance = (productNumber === 1) ? sentimentChart1 : sentimentChart2;
    const chartInstance = displayChart(data.sentiment_stats, chartCanvasId, existingChartInstance);
    chartInstanceVarSetter(chartInstance);

    displayComments(data.representative_comments || {}, currentCommentsSection);
    displayRecommendation(data.recommendation || {}, currentRecResult, currentRecReason);
}

function truncateUrl(url, maxLength = 50) {
    if(!url) return "";
    if (url.length <= maxLength) return url;
    return url.substring(0, maxLength - 3) + "...";
}


function displayChart(sentiments, canvasId, existingChartInstance) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    if (existingChartInstance) {
        existingChartInstance.destroy();
    }

    const total = (sentiments["Tích cực"] || 0) + (sentiments["Trung bình"] || 0) + (sentiments["Tiêu cực"] || 0);

    const dataCounts = [
        sentiments["Tiêu cực"] || 0,
        sentiments["Trung bình"] || 0,
        sentiments["Tích cực"] || 0,
    ];

    const dataPercentages = dataCounts.map(count => {
        return total > 0 ? ((count / total) * 100).toFixed(1) : "0.0";
    });

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Tiêu cực', 'Trung bình', 'Tích cực'],
            datasets: [{
                label: 'Số lượng',
                data: dataCounts,
                backgroundColor: [
                    'rgba(220, 53, 69, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(40, 167, 69, 0.8)',
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    suggestedMax: total > 0 ? Math.max(10, ...dataCounts) * 1.2 : 10,
                    title: { display: true, text: 'Số lượng' }
                },
                y: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const count = context.parsed.x;
                            const percentage = dataPercentages[context.dataIndex];
                            return `${count} (${percentage}%)`;
                        }
                    }
                },
                datalabels: {
                    anchor: 'end',
                    align: 'end',
                    formatter: function(value, context) {
                        const percentage = dataPercentages[context.dataIndex];
                        if (value === 0 && parseFloat(percentage) === 0) return '';
                        return `${value}\n(${percentage}%)`;
                    },
                    color: '#fff',
                    font: { weight: 'bold', size: 10 },
                    textStrokeColor: '#000',
                    textStrokeWidth: 2,
                    textAlign: 'center'
                }
            }
        },
        plugins: [ChartDataLabels]
    });
}

function displayComments(comments, sectionElement) {
    const commentTypes = [
        { key: 'negative', title: '😞 Bình luận tiêu cực nổi bật', class: 'negative' },
        { key: 'neutral', title: '😐 Bình luận trung bình nổi bật', class: 'neutral' },
        { key: 'positive', title: '😊 Bình luận tích cực nổi bật', class: 'positive' },
    ];

    sectionElement.innerHTML = '';
    let hasComments = false;

    commentTypes.forEach(type => {
        const typeComments = comments[type.key] || [];
        if (typeComments.length > 0) {
            hasComments = true;
            const commentGroup = document.createElement('div');
            commentGroup.className = `comment-group ${type.class}`;

            commentGroup.innerHTML = `
                <h4>${type.title}</h4>
                ${typeComments.slice(0, Config.MAX_REPRESENTATIVE_COMMENTS || 5).map(comment =>
                    `<div class="comment-item">"${escapeHtml(comment)}"</div>`
                ).join('')}
            `;
            sectionElement.appendChild(commentGroup);
        }
    });
    if (!hasComments) {
        sectionElement.innerHTML = '<p class="no-comments">Không có bình luận đại diện nào để hiển thị.</p>';
    }
}

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return '';
    return unsafe
         .replace(/&/g, "&")
         .replace(/</g, "<")
         .replace(/>/g, ">")
         .replace(/"/g, "\"")
         .replace(/'/g, "'");
}

function displayRecommendation(recommendation, resultElement, reasonElement) {
    const decision = recommendation.decision || 'consider';
    const reason = recommendation.reason || 'Không có đủ thông tin để đưa ra khuyến nghị.';
    let resultClass, resultText, icon;

    switch (decision) {
        case 'buy': resultClass = 'recommend-buy'; resultText = 'Nên mua'; icon = '👍'; break;
        case 'avoid': resultClass = 'recommend-avoid'; resultText = 'Không nên mua'; icon = '👎'; break;
        default: resultClass = 'recommend-consider'; resultText = 'Cân nhắc kỹ'; icon = '🤔';
    }
    resultElement.className = `recommendation-result ${resultClass}`;
    resultElement.innerHTML = `${icon} ${resultText}`;
    reasonElement.textContent = reason;
}

function performComparison(data1, data2) {
    const stats1 = data1.sentiment_stats;
    const total1 = data1.total_reviews || ((stats1["Tích cực"] || 0) + (stats1["Trung bình"] || 0) + (stats1["Tiêu cực"] || 0));
    const positivePercentage1 = total1 > 0 ? ((stats1["Tích cực"] || 0) / total1 * 100) : 0;
    const confidence1 = data1.recommendation.confidence || 0;

    const stats2 = data2.sentiment_stats;
    const total2 = data2.total_reviews || ((stats2["Tích cực"] || 0) + (stats2["Trung bình"] || 0) + (stats2["Tiêu cực"] || 0));
    const positivePercentage2 = total2 > 0 ? ((stats2["Tích cực"] || 0) / total2 * 100) : 0;
    const confidence2 = data2.recommendation.confidence || 0;

    let comparisonMessage = "";
    const score1 = (positivePercentage1 * 0.7) + (confidence1 * (data1.recommendation.decision === 'buy' ? 30 : (data1.recommendation.decision === 'consider' ? 15 : 0) ));
    const score2 = (positivePercentage2 * 0.7) + (confidence2 * (data2.recommendation.decision === 'buy' ? 30 : (data2.recommendation.decision === 'consider' ? 15 : 0) ));
    const diffThreshold = 5; // Difference threshold for scores

    if (score1 > score2 + diffThreshold) {
        comparisonMessage = `Nên ưu tiên <strong>Sản phẩm 1</strong>. <br>Sản phẩm này có điểm đánh giá tổng hợp (${score1.toFixed(1)}) cao hơn đáng kể so với Sản phẩm 2 (${score2.toFixed(1)}), dựa trên tỷ lệ tích cực và độ tin cậy khuyến nghị.`;
    } else if (score2 > score1 + diffThreshold) {
        comparisonMessage = `Nên ưu tiên <strong>Sản phẩm 2</strong>. <br>Sản phẩm này có điểm đánh giá tổng hợp (${score2.toFixed(1)}) cao hơn đáng kể so với Sản phẩm 1 (${score1.toFixed(1)}), dựa trên tỷ lệ tích cực và độ tin cậy khuyến nghị.`;
    } else {
         comparisonMessage = `Cả hai sản phẩm có điểm đánh giá tổng hợp khá tương đồng (SP1: ${score1.toFixed(1)}, SP2: ${score2.toFixed(1)}). <br>Hãy xem xét kỹ các bình luận chi tiết, tính năng cụ thể của từng sản phẩm và nhu cầu cá nhân để đưa ra lựa chọn tốt nhất.`;
    }
    comparisonResultText.innerHTML = comparisonMessage;

    comparisonProd1Stats.innerHTML = `
        URL: <a href="${data1.product_url}" target="_blank" rel="noopener noreferrer">${truncateUrl(data1.product_url, 30)}</a><br>
        Tổng đánh giá: ${total1}<br>
        Tích cực: ${stats1["Tích cực"] || 0} (${positivePercentage1.toFixed(1)}%)<br>
        Trung bình: ${stats1["Trung bình"] || 0}<br>
        Tiêu cực: ${stats1["Tiêu cực"] || 0}<br>
        Khuyến nghị: ${data1.recommendation.decision} (Độ tin cậy: ${(confidence1 * 100).toFixed(1)}%)
    `;
    comparisonProd2Stats.innerHTML = `
        URL: <a href="${data2.product_url}" target="_blank" rel="noopener noreferrer">${truncateUrl(data2.product_url, 30)}</a><br>
        Tổng đánh giá: ${total2}<br>
        Tích cực: ${stats2["Tích cực"] || 0} (${positivePercentage2.toFixed(1)}%)<br>
        Trung bình: ${stats2["Trung bình"] || 0}<br>
        Tiêu cực: ${stats2["Tiêu cực"] || 0}<br>
        Khuyến nghị: ${data2.recommendation.decision} (Độ tin cậy: ${(confidence2 * 100).toFixed(1)}%)
    `;
}


function clearAllResults() {
    errorSection.style.display = 'none';
    errorSection.innerHTML = '';

    resultsContainer.style.display = 'none'; // Hide the main container for all results
    resultsSection1.style.display = 'none';
    resultsSection2.style.display = 'none';
    comparisonSection.style.display = 'none';


    if (sentimentChart1) { sentimentChart1.destroy(); sentimentChart1 = null; }
    if (sentimentChart2) { sentimentChart2.destroy(); sentimentChart2 = null; }

    commentsSection1.innerHTML = '';
    recommendationResult1.innerHTML = '';
    recommendationReason1.textContent = '';
    productTitle1.textContent = currentMode === 'single' ? 'Kết quả Phân tích' : 'Kết quả cho Sản phẩm 1';


    commentsSection2.innerHTML = '';
    recommendationResult2.innerHTML = '';
    recommendationReason2.textContent = '';
    productTitle2.textContent = 'Kết quả cho Sản phẩm 2';


    comparisonResultText.innerHTML = '';
    comparisonProd1Stats.innerHTML = '';
    comparisonProd2Stats.innerHTML = '';
}

function showError(message) {
    errorSection.innerHTML = message; // Use innerHTML to allow <br>
    errorSection.style.display = 'block';
}

const Config = {
    MAX_REPRESENTATIVE_COMMENTS: 5
};