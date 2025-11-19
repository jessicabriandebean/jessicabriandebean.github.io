import React, { useState } from 'react';
import { Download, Edit3, Briefcase, GraduationCap, Award, Code, TrendingUp } from 'lucide-react';

const ResumeTemplate = () => {
  const [editing, setEditing] = useState(false);
  const [resumeData, setResumeData] = useState({
    name: "Jessica Bean",
    title: "Data Scientist | Federal Government → Private Sector",
    contact: {
      email: "jessicabbean@gmail.com",
      phone: "(206) 591-0312",
      linkedin: "linkedin.com/in/jessicabbean",
      github: "github.com/jessicabriandebean",
      portfolio: "yourportfolio.github.io" 
    },
    summary: "Results-driven Data Scientist with 10 years of experience leveraging advanced analytics and machine learning to drive policy and operational decisions in the federal government. Proven track record of working with large-scale datasets, ensuring regulatory compliance, and communicating complex insights to diverse stakeholders. Seeking to apply technical expertise and ethical data practices to solve business challenges in [target industry]. Security clearance: [level, if applicable].",
    skills: {
      programming: ["Python", "R", "SQL"],
      ml: ["Scikit-learn", "TensorFlow", "PyTorch", "XGBoost"],
      visualization: ["Tableau", "Power BI", "Plotly"],
      cloud: ["AWS", "Azure"],
      tools: ["Git", "Docker", "Jupyter", "Apache Spark"],
      statistics: ["A/B Testing", "Causal Inference", "Time Series", "Bayesian Methods"]
    },
    experience: [
      {
        title: "Data Scientist",
        org: "Department of Defense",
        location: "Keyport, Washington",
        date: "January 2020 - September 2025",
        achievements: [
          "Developed machine learning models that improved supply chain efficiency by 85%, resulting in $10 million in cost savings",
          "Led cross-functional team of 3 analysts to build automated reporting dashboard serving U.S. Navy stakeholders, reducing manual reporting time by %",
          "Designed and implemented distributed data architecture and framework that optimized data analytics across the Navy, increasing user engagement by 70%",
          "Collaborated with policy teams to translate complex statistical findings into actionable recommendations for senior leadership",
          "Ensured GDRP compliance and data privacy and ethical standards across all analytical projects"
        ]80
      },
      {
        title: "[Data Analytics Lead]",
        org: "[Department of Defense]",
        location: "Keyport, Washington",
        date: "June 2017 - December 2019",
        achievements: [
          "Built predictive models using time-series, regression, and machine learning techniques that forecasted obsolescence dates with 70% accuracy",
          "Processed and analyzed datasets containing 7 million records using distributed computing",
          "Presented findings to executive leadership, influencing strategic decisions worth $5 million"
        ]
      }
    ],
    projects: [
      {
        name: "Economic Indicator Forecasting Platform",
        tech: "Python, Prophet, Streamlit, AWS",
        description: "Built time series forecasting tool predicting unemployment and inflation trends with interactive visualizations",
        link: "github.com/username/project1"
      },
      {
        name: "Algorithmic Bias Detection Toolkit",
        tech: "Python, Fairlearn, Scikit-learn, FastAPI",
        description: "Developed open-source toolkit to identify and mitigate bias in ML models with automated fairness metrics",
        link: "github.com/username/project2"
      },
      {
        name: "Customer Segmentation Engine",
        tech: "Python, K-Means, RFM Analysis, Plotly",
        description: "Created end-to-end ML pipeline for customer segmentation with ROI calculator and interactive dashboard",
        link: "github.com/username/project3"
      }
    ],
    education: [
      {
        degree: "Master of Science in Data Science / Statistics / Operations",
        school: "Pennsylvania State University",
        date: "2019",
        details: "Relevant coursework: Machine Learning, Statistical Inference, Data Mining"
      },
      {
        degree: "Bachelor of Science in Business Analytics",
        school: "California State University, Fullerton",
        date: "2015"
      }
    ],
    certifications: [
      "AWS Certified Machine Learning - Specialty (or relevant cert)",
      "Certified Analytics Professional (CAP)",
      "Security Clearance: [Level] (if applicable)"
    ]
  });

  return (
    <div className="max-w-4xl mx-auto p-8 bg-white">
      {/* Header */}
      <div className="text-center border-b-2 border-gray-800 pb-4 mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-1">{resumeData.name}</h1>
        <p className="text-lg text-gray-700 mb-2">{resumeData.title}</p>
        <div className="flex flex-wrap justify-center gap-3 text-sm text-gray-600">
          <span>{resumeData.contact.email}</span>
          <span>•</span>
          <span>{resumeData.contact.phone}</span>
          <span>•</span>
          <span>{resumeData.contact.linkedin}</span>
          <span>•</span>
          <span>{resumeData.contact.github}</span>
          <span>•</span>
          <span>{resumeData.contact.portfolio}</span>
        </div>
      </div>

      {/* Professional Summary */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2 flex items-center gap-2">
          <Briefcase size={20} />
          PROFESSIONAL SUMMARY
        </h2>
        <p className="text-sm text-gray-800 leading-relaxed">{resumeData.summary}</p>
      </section>

      {/* Technical Skills */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2 flex items-center gap-2">
          <Code size={20} />
          TECHNICAL SKILLS
        </h2>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="font-semibold">Programming:</span> {resumeData.skills.programming.join(", ")}
          </div>
          <div>
            <span className="font-semibold">ML/AI:</span> {resumeData.skills.ml.join(", ")}
          </div>
          <div>
            <span className="font-semibold">Visualization:</span> {resumeData.skills.visualization.join(", ")}
          </div>
          <div>
            <span className="font-semibold">Cloud:</span> {resumeData.skills.cloud.join(", ")}
          </div>
          <div>
            <span className="font-semibold">Tools:</span> {resumeData.skills.tools.join(", ")}
          </div>
          <div>
            <span className="font-semibold">Statistics:</span> {resumeData.skills.statistics.join(", ")}
          </div>
        </div>
      </section>

      {/* Professional Experience */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2 flex items-center gap-2">
          <TrendingUp size={20} />
          PROFESSIONAL EXPERIENCE
        </h2>
        {resumeData.experience.map((job, idx) => (
          <div key={idx} className="mb-4">
            <div className="flex justify-between items-start mb-1">
              <div>
                <h3 className="font-bold text-gray-900">{job.title}</h3>
                <p className="text-sm text-gray-700">{job.org}, {job.location}</p>
              </div>
              <span className="text-sm text-gray-600">{job.date}</span>
            </div>
            <ul className="list-disc list-outside ml-5 text-sm text-gray-800 space-y-1">
              {job.achievements.map((achievement, i) => (
                <li key={i}>{achievement}</li>
              ))}
            </ul>
          </div>
        ))}
      </section>

      {/* Featured Projects */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2 flex items-center gap-2">
          <Award size={20} />
          FEATURED DATA SCIENCE PROJECTS
        </h2>
        {resumeData.projects.map((project, idx) => (
          <div key={idx} className="mb-3">
            <div className="flex justify-between items-start">
              <h3 className="font-bold text-gray-900">{project.name}</h3>
              <span className="text-xs text-blue-600">{project.link}</span>
            </div>
            <p className="text-xs text-gray-600 mb-1">{project.tech}</p>
            <p className="text-sm text-gray-800">{project.description}</p>
          </div>
        ))}
      </section>

      {/* Education */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2 flex items-center gap-2">
          <GraduationCap size={20} />
          EDUCATION
        </h2>
        {resumeData.education.map((edu, idx) => (
          <div key={idx} className="mb-2">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-bold text-gray-900">{edu.degree}</h3>
                <p className="text-sm text-gray-700">{edu.school}</p>
                {edu.details && <p className="text-xs text-gray-600">{edu.details}</p>}
              </div>
              <span className="text-sm text-gray-600">{edu.date}</span>
            </div>
          </div>
        ))}
      </section>

      {/* Certifications */}
      <section className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 border-b border-gray-400 mb-2">
          CERTIFICATIONS & CLEARANCES
        </h2>
        <ul className="list-disc list-outside ml-5 text-sm text-gray-800 space-y-1">
          {resumeData.certifications.map((cert, idx) => (
            <li key={idx}>{cert}</li>
          ))}
        </ul>
      </section>

      {/* Instructions */}
      <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded">
        <h3 className="font-bold text-blue-900 mb-2">📝 Customization Instructions:</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Replace ALL placeholder text with your actual information</li>
          <li>• Quantify achievements with numbers (%, $, time saved, users impacted)</li>
          <li>• Use action verbs: Built, Developed, Led, Optimized, Designed, Implemented</li>
          <li>• Tailor the summary for each industry (finance, marketing, etc.)</li>
          <li>• Keep to 2 pages maximum for resume, 1 page for one-pager version</li>
          <li>• Remove security clearance info if not relevant to position</li>
          <li>• For one-pager: Remove Featured Projects section, condense experience</li>
        </ul>
      </div>
    </div>
  );
};

export default ResumeTemplate;
