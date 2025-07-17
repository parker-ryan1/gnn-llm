import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from .web_scraper import WebScraper, Product
from .order_automation import OrderAutomation, OrderRequest, OrderResult
from rag.rag_system import OllamaRAGSystem

@dataclass
class PurchaseRecommendation:
    product: Product
    confidence_score: float
    reasoning: str
    gnn_relevance: float
    llm_recommendation: str

@dataclass
class OrderingConfig:
    max_budget: float = 1000.0
    max_items: int = 5
    min_confidence: float = 0.7
    preferred_sites: List[str] = None
    auto_order: bool = False  # Set to True to actually place orders
    
    def __post_init__(self):
        if self.preferred_sites is None:
            self.preferred_sites = ["amazon", "ebay"]

class IntelligentOrderingSystem:
    def __init__(self, rag_system: OllamaRAGSystem, ordering_config: OrderingConfig):
        self.rag_system = rag_system
        self.config = ordering_config
        self.web_scraper = WebScraper(headless=True)
        self.order_automation = OrderAutomation(headless=False)  # Non-headless for demo
        self.logger = logging.getLogger(__name__)
    
    def analyze_gnn_results_for_purchasing(self, gnn_results: Dict[str, Any], 
                                         original_data: Any) -> List[str]:
        """Analyze GNN results to determine what products to search for"""
        try:
            # Use RAG to interpret GNN results and suggest products
            analysis_prompt = f"""
            Based on the following graph neural network analysis results, suggest specific products 
            that would be relevant to purchase. Focus on items that could help with the patterns 
            or relationships identified in the data.
            
            GNN Results Summary:
            - Number of nodes: {gnn_results.get('graph', {}).get('number_of_nodes', 'N/A')}
            - Training loss: {gnn_results.get('training_losses', ['N/A'])[-1] if gnn_results.get('training_losses') else 'N/A'}
            - Data categories found: {self._extract_categories_from_data(original_data)}
            
            Please suggest 3-5 specific product names that would be useful based on this analysis.
            Format your response as a simple list of product names, one per line.
            """
            
            rag_response = self.rag_system.rag_query(analysis_prompt)
            
            # Extract product suggestions from the response
            product_suggestions = self._parse_product_suggestions(rag_response['response'])
            
            self.logger.info(f"Generated {len(product_suggestions)} product suggestions from GNN analysis")
            return product_suggestions
            
        except Exception as e:
            self.logger.error(f"Error analyzing GNN results: {e}")
            # Fallback suggestions
            return ["laptop", "wireless mouse", "notebook", "desk organizer", "monitor"]
    
    def _extract_categories_from_data(self, data: Any) -> List[str]:
        """Extract categories or themes from the original data"""
        categories = []
        try:
            if hasattr(data, 'columns') and 'category' in data.columns:
                categories = data['category'].unique().tolist()
            elif isinstance(data, dict):
                categories = list(data.keys())
        except:
            pass
        return categories[:5]  # Limit to 5 categories
    
    def _parse_product_suggestions(self, llm_response: str) -> List[str]:
        """Parse product suggestions from LLM response"""
        suggestions = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            # Remove bullet points, numbers, etc.
            if line and not line.startswith('#'):
                # Clean up the line
                clean_line = line.replace('-', '').replace('*', '').replace('•', '')
                clean_line = clean_line.strip()
                if len(clean_line) > 2 and len(clean_line) < 50:
                    suggestions.append(clean_line)
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def search_and_evaluate_products(self, product_queries: List[str]) -> List[PurchaseRecommendation]:
        """Search for products and evaluate them using LLM"""
        recommendations = []
        
        for query in product_queries:
            self.logger.info(f"Searching for: {query}")
            
            # Search on preferred sites
            all_products = []
            for site in self.config.preferred_sites:
                try:
                    products = self.web_scraper.search_products(query, site)
                    all_products.extend(products)
                except Exception as e:
                    self.logger.warning(f"Search failed on {site}: {e}")
            
            # Evaluate each product
            for product in all_products[:3]:  # Limit to top 3 per query
                try:
                    recommendation = self._evaluate_product(product, query)
                    if recommendation.confidence_score >= self.config.min_confidence:
                        recommendations.append(recommendation)
                except Exception as e:
                    self.logger.warning(f"Product evaluation failed: {e}")
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        return recommendations[:self.config.max_items]
    
    def _evaluate_product(self, product: Product, original_query: str) -> PurchaseRecommendation:
        """Evaluate a product using LLM and assign confidence score"""
        evaluation_prompt = f"""
        Please evaluate this product for purchase based on the original search query:
        
        Original Query: {original_query}
        Product Name: {product.name}
        Price: ${product.price}
        Rating: {product.rating}/5
        Available: {product.availability}
        
        Please provide:
        1. A confidence score from 0.0 to 1.0 for how well this matches the query
        2. A brief explanation of why this product is or isn't a good match
        3. Any concerns about the price or quality
        
        Format your response as:
        CONFIDENCE: [score]
        REASONING: [explanation]
        """
        
        try:
            rag_response = self.rag_system.rag_query(evaluation_prompt)
            response_text = rag_response['response']
            
            # Parse confidence score
            confidence_score = self._extract_confidence_score(response_text)
            
            # Extract reasoning
            reasoning = self._extract_reasoning(response_text)
            
            # Calculate GNN relevance (simplified - based on price and rating)
            gnn_relevance = self._calculate_gnn_relevance(product)
            
            return PurchaseRecommendation(
                product=product,
                confidence_score=confidence_score,
                reasoning=reasoning,
                gnn_relevance=gnn_relevance,
                llm_recommendation=response_text
            )
            
        except Exception as e:
            self.logger.error(f"Product evaluation failed: {e}")
            return PurchaseRecommendation(
                product=product,
                confidence_score=0.5,
                reasoning="Evaluation failed",
                gnn_relevance=0.5,
                llm_recommendation="Error in evaluation"
            )
    
    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from LLM response"""
        try:
            lines = response.split('\n')
            for line in lines:
                if 'CONFIDENCE:' in line.upper():
                    score_text = line.split(':')[1].strip()
                    return float(score_text)
        except:
            pass
        
        # Fallback: try to find any number between 0 and 1
        import re
        numbers = re.findall(r'0\.\d+|1\.0', response)
        if numbers:
            return float(numbers[0])
        
        return 0.5  # Default
    
    def _extract_reasoning(self, response: str) -> str:
        """Extract reasoning from LLM response"""
        try:
            lines = response.split('\n')
            for i, line in enumerate(lines):
                if 'REASONING:' in line.upper():
                    reasoning_lines = lines[i:]
                    return ' '.join(reasoning_lines).replace('REASONING:', '').strip()
        except:
            pass
        
        return response[:200]  # Fallback to first 200 chars
    
    def _calculate_gnn_relevance(self, product: Product) -> float:
        """Calculate relevance score based on GNN-style metrics"""
        score = 0.5  # Base score
        
        # Factor in price (prefer reasonable prices)
        if 10 <= product.price <= 500:
            score += 0.2
        elif product.price > 500:
            score -= 0.1
        
        # Factor in rating
        if product.rating >= 4.0:
            score += 0.2
        elif product.rating >= 3.0:
            score += 0.1
        
        # Factor in availability
        if product.availability:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def execute_intelligent_ordering(self, gnn_results: Dict[str, Any], 
                                   original_data: Any,
                                   user_credentials: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute the complete intelligent ordering process"""
        results = {
            'product_suggestions': [],
            'recommendations': [],
            'orders_placed': [],
            'total_cost': 0.0,
            'success': False,
            'error_messages': []
        }
        
        try:
            # Step 1: Analyze GNN results to get product suggestions
            self.logger.info("Analyzing GNN results for product suggestions...")
            product_suggestions = self.analyze_gnn_results_for_purchasing(gnn_results, original_data)
            results['product_suggestions'] = product_suggestions
            
            # Step 2: Search and evaluate products
            self.logger.info("Searching and evaluating products...")
            recommendations = self.search_and_evaluate_products(product_suggestions)
            results['recommendations'] = [
                {
                    'product_name': rec.product.name,
                    'price': rec.product.price,
                    'confidence_score': rec.confidence_score,
                    'reasoning': rec.reasoning,
                    'url': rec.product.url
                }
                for rec in recommendations
            ]
            
            # Step 3: Filter by budget
            budget_filtered = self._filter_by_budget(recommendations)
            
            if not budget_filtered:
                results['error_messages'].append("No products within budget")
                return results
            
            # Step 4: Place orders (if auto_order is enabled)
            if self.config.auto_order and user_credentials:
                self.logger.info("Placing orders...")
                order_results = self._place_orders(budget_filtered, user_credentials)
                results['orders_placed'] = order_results
                results['total_cost'] = sum(order.total_cost for order in order_results if order.success)
            else:
                self.logger.info("Auto-ordering disabled - showing recommendations only")
                results['total_cost'] = sum(rec.product.price for rec in budget_filtered)
            
            results['success'] = True
            
        except Exception as e:
            self.logger.error(f"Intelligent ordering failed: {e}")
            results['error_messages'].append(str(e))
        
        return results
    
    def _filter_by_budget(self, recommendations: List[PurchaseRecommendation]) -> List[PurchaseRecommendation]:
        """Filter recommendations by budget constraints"""
        filtered = []
        total_cost = 0.0
        
        for rec in recommendations:
            if total_cost + rec.product.price <= self.config.max_budget:
                filtered.append(rec)
                total_cost += rec.product.price
            
            if len(filtered) >= self.config.max_items:
                break
        
        return filtered
    
    def _place_orders(self, recommendations: List[PurchaseRecommendation], 
                     credentials: Dict[str, str]) -> List[OrderResult]:
        """Place orders for recommended products (DEMO MODE)"""
        order_results = []
        
        # Group by site
        site_products = {}
        for rec in recommendations:
            if "amazon.com" in rec.product.url:
                site = "amazon"
            elif "ebay.com" in rec.product.url:
                site = "ebay"
            else:
                continue
            
            if site not in site_products:
                site_products[site] = []
            site_products[site].append(rec.product)
        
        # Place orders per site
        for site, products in site_products.items():
            try:
                # Login to site
                if self.order_automation.login_to_site(site, credentials):
                    
                    # Add products to cart
                    for product in products:
                        self.order_automation.add_to_cart(product)
                    
                    # Create order request (DEMO - don't include real payment info)
                    order_request = OrderRequest(
                        products=products,
                        shipping_address={},  # Empty for demo
                        payment_info={},      # Empty for demo
                        user_credentials=credentials
                    )
                    
                    # Proceed to checkout (DEMO MODE)
                    order_result = self.order_automation.proceed_to_checkout(order_request)
                    order_results.append(order_result)
                    
            except Exception as e:
                self.logger.error(f"Order failed for {site}: {e}")
                order_results.append(OrderResult(success=False, error_message=str(e)))
        
        return order_results
    
    def close(self):
        """Clean up resources"""
        self.web_scraper.close()
        self.order_automation.close()