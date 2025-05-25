        // DOM Elements
        const productUrlInput = document.getElementById('productUrl');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const loadingDiv = document.getElementById('loading');
        const errorSection = document.getElementById('errorSection');
        const resultsSection = document.getElementById('resultsSection');
        const productName = document.getElementById('productName');
        const productStats = document.getElementById('productStats');
        const commentsSection = document.getElementById('commentsSection');
        const recommendationResult = document.getElementById('recommendationResult');
        const recommendationReason = document.getElementById('recommendationReason');
        
        let sentimentChart = null;



        // Event Listeners
        productUrlInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                analyzeProduct();
            }
        });

        // Main Analysis Function
        async function analyzeProduct() {
            const url = productUrlInput.value.trim();
            
            // Reset UI
            clearResults();
            
            // if (!url) {
            //     showError('Vui lòng nhập link sản phẩm từ Điện máy xanh.');
            //     return;
            // }
            
            // if (!url.includes('dienmayxanh.com')) {
            //     showError('Vui lòng nhập link hợp lệ từ website Điện máy xanh.');
            //     return;
            // }
            
            // Show loading
            loadingDiv.style.display = 'block';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/analyze-product', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url }),
                    signal: AbortSignal.timeout(1200000) // 10 minutes timeout
                });
                
                if (!response.ok) {
                    let errorMsg = `Lỗi ${response.status}: `;
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
                    throw new Error(data.error);
                } else {
                    displayResults(data);
                    resultsSection.style.display = 'block';
                }
                
            } catch (error) {
                console.error('Analysis Error:', error);
                let displayError = 'Đã xảy ra lỗi khi phân tích sản phẩm. Vui lòng thử lại.';
                if (error.name === 'AbortError') {
                    displayError = 'Yêu cầu đã hết thời gian chờ. Vui lòng thử lại.';
                } else if (error.message) {
                    displayError = error.message;
                }
                showError(displayError);
            } finally {
                loadingDiv.style.display = 'none';
                analyzeBtn.disabled = false;
            }
        }

        // Display Results
        function displayResults(data) {

            // Chart
            displayChart(data.sentiment_stats);
            
            // Representative Comments
            displayComments(data.representative_comments || {});
            
            // Recommendation
            displayRecommendation(data.recommendation || {});
        }

        // Display Bar Chart with Count and Percentage
        function displayChart(sentiments) {
            const ctx = document.getElementById('sentimentChart').getContext('2d');

            if (sentimentChart) {
                sentimentChart.destroy();
            }

            const total = sentiments["Tích cực"] + sentiments["Trung bình"] + sentiments["Tiêu cực"];

            const dataCounts = [
                sentiments["Tiêu cực"],
                sentiments["Trung bình"],
                sentiments["Tích cực"],
                
            ];

            const dataPercentages = dataCounts.map(count => {
                return total > 0 ? ((count / total) * 100).toFixed(1) : 0;
            });

            sentimentChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Tiêu cực', 'Trung bình','Tích cực' ],
                    datasets: [{
                        label: 'Số lượng',
                        data: dataCounts,
                        backgroundColor: [
                            'rgba(220, 53, 69, 0.8)',
                            'rgba(255, 193, 7, 0.8)',
                            'rgba(40, 167, 69, 0.8)',
                            
                        ],
                        // borderColor: [
                        //     'rgba(40, 167, 69, 1)',
                        //     'rgba(255, 193, 7, 1)',
                        //     'rgba(220, 53, 69, 1)'
                        // ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            suggestedMax: Math.max(...dataCounts) * 1.3, 
                            title: {
                                display: true,
                                text: 'Số lượng'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const count = context.parsed.y;
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
                                return `${value} (${percentage}%)`;
                            },
                            color: '#000',
                            font: {
                                weight: 'bold'
                            }
                        }
                    }
                },
                plugins: [ChartDataLabels]
            });
        }

        function displayComments(comments) {
            const commentTypes = [
                { key: 'negative', title: '😞 Bình luận tiêu cực', class: 'negative' },
                { key: 'neutral', title: '😐 Bình luận trung bình', class: 'neutral' },
                { key: 'positive', title: '😊 Bình luận tích cực', class: 'positive' },
            ];
            
            commentsSection.innerHTML = '';
            
            commentTypes.forEach(type => {
                const typeComments = comments[type.key] || [];
                if (typeComments.length > 0) {
                    const commentGroup = document.createElement('div');
                    commentGroup.className = `comment-group ${type.class}`;
                    
                    commentGroup.innerHTML = `
                        <h4>${type.title}</h4>
                        ${typeComments.slice(0, 5).map(comment => 
                            `<div class="comment-item">"${comment}"</div>`
                        ).join('')}
                    `;
                    
                    commentsSection.appendChild(commentGroup);
                }
            });
        }

        // Display Recommendation
        function displayRecommendation(recommendation) {
            const decision = recommendation.decision || 'consider';
            const reason = recommendation.reason || 'Không có đủ thông tin để đưa ra khuyến nghị.';
            
            let resultClass, resultText, icon;
            
            switch (decision) {
                case 'buy':
                    resultClass = 'recommend-buy';
                    resultText = 'Nên mua';
                    icon = '👍';
                    break;
                case 'avoid':
                    resultClass = 'recommend-avoid';
                    resultText = 'Không nên mua';
                    icon = '👎';
                    break;
                default:
                    resultClass = 'recommend-consider';
                    resultText = 'Cân nhắc kỹ';
                    icon = '🤔 ';
            }
            
            recommendationResult.className = `recommendation-result ${resultClass}`;
            recommendationResult.innerHTML = `${icon} ${resultText}`;
            recommendationReason.textContent = reason;
        }

        // Helper Functions
        function clearResults() {
            errorSection.style.display = 'none';
            resultsSection.style.display = 'none';
            errorSection.textContent = '';
            
            if (sentimentChart) {
                sentimentChart.destroy();
                sentimentChart = null;
            }
        }

        function showError(message) {
            errorSection.textContent = message;
            errorSection.style.display = 'block';
            resultsSection.style.display = 'none';
        }
 