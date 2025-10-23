/**
 * Mock AI Risk Assessment Model
 * In production, this would call a real ML model API
 */

export interface RiskAssessmentResult {
  riskScore: 'low' | 'medium' | 'high';
  confidence: number;
  findings: string[];
}

/**
 * Simulates an AI model analyzing a medical image
 * @param imageData Base64 encoded image data
 * @returns Promise with risk assessment result
 */
export async function analyzeImageRisk(imageData: string): Promise<RiskAssessmentResult> {
  // Validate input
  if (!imageData || imageData.length < 100) {
    throw new Error('Invalid image data');
  }

  // Create a timeout promise to prevent hanging
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error('Analysis timeout - please try again')), 10000);
  });

  // Simulate API call with timeout protection
  const analysisPromise = new Promise<void>(resolve => {
    setTimeout(resolve, 500 + Math.random() * 300);
  });

  // Race between analysis and timeout
  await Promise.race([analysisPromise, timeoutPromise]);

  // Mock analysis: Generate deterministic result based on image data
  // In production, this would send the image to a real AI model
  const hash = simpleHash(imageData);
  const randomValue = hash % 100;

  let riskScore: 'low' | 'medium' | 'high';
  let confidence: number;
  let findings: string[];

  if (randomValue < 60) {
    // 60% chance of low risk
    riskScore = 'low';
    confidence = 0.85 + Math.random() * 0.1;
    findings = [
      'No acute abnormalities detected',
      'Normal cardiac silhouette',
      'Clear lung fields',
    ];
  } else if (randomValue < 85) {
    // 25% chance of medium risk
    riskScore = 'medium';
    confidence = 0.75 + Math.random() * 0.15;
    findings = [
      'Mild opacity detected in lower lobe',
      'Possible early signs of inflammation',
      'Recommend follow-up in 2-4 weeks',
    ];
  } else {
    // 15% chance of high risk
    riskScore = 'high';
    confidence = 0.82 + Math.random() * 0.12;
    findings = [
      'Significant abnormality detected',
      'Dense consolidation observed',
      'Immediate physician consultation recommended',
    ];
  }

  return {
    riskScore,
    confidence: Math.round(confidence * 100) / 100,
    findings,
  };
}

/**
 * Simple hash function for deterministic results
 */
function simpleHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < Math.min(str.length, 1000); i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}