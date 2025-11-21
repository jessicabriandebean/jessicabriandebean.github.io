import React, { useState } from 'react';
import { CheckCircle, Circle, ChevronRight, ExternalLink, Github, Globe, FileText } from 'lucide-react';

const SetupGuide = () => {
  const [completedSteps, setCompletedSteps] = useState({});
  const [activeSection, setActiveSection] = useState('portfolio');

  const toggleStep = (sectionId, stepId) => {
    setCompletedSteps(prev => ({
      ...prev,
      [`${sectionId}-${stepId}`]: !prev[`${sectionId}-${stepId}`]
    }));
  };

  const sections = [
    {
      id: 'portfolio',
      title: '🌐 Portfolio Website Setup',
      icon: Globe,
      color: 'blue',
      steps: [
        {
          id: 1,
          title: 'Create GitHub Account',
          description: 'Sign up at github.com with a professional username',
          details: [
            'Go to https://github.com',
            'Click "Sign up"',
            'Choose username: yourname-data or yourname',
            'Use professional email',
            'Verify email address'
          ],
          url: 'https://github.com/signup'
        },
        {
          id: 2,
          title: 'Create Portfolio Repository',
          description: 'Create repository named exactly yourusername.github.io',
          details: [
            'Click "+" icon → New repository',
            'Name: yourusername.github.io (exact!)',
            'Description: "Data Science Portfolio"',
            'Public, Add README, Choose license',
            'Click Create repository'
          ]
        },
        {
          id: 3,
          title: 'Create Folder Structure',
          description: 'Set up organized folders for your content',
          details: [
            'Click "Add file" → "Create new file"',
            'Type: blog/.gitkeep (slash creates folder)',
            'Repeat for: images/, css/, js/, assets/',
            'Commit each file'
          ]
        },
        {
          id: 4,
          title: 'Upload Portfolio Homepage',
          description: 'Add your index.html file',
          details: [
            'Save the portfolio HTML as index.html',
            'In repo: "Add file" → "Upload files"',
            'Drag index.html or select it',
            'Commit message: "Add portfolio homepage"',
            'Click Commit changes'
          ]
        },
        {
          id: 5,
          title: 'Enable GitHub Pages',
          description: 'Make your site live on the internet',
          details: [
            'Go to Settings tab',
            'Click "Pages" in left sidebar',
            'Source: Branch = main, Folder = / (root)',
            'Click Save',
            'Wait 2-5 minutes for deployment',
            'Visit https://yourusername.github.io'
          ]
        },
        {
          id: 6,
          title: 'Customize Your Content',
          description: 'Replace placeholder text with your information',
          details: [
            'Click index.html → Edit (pencil icon)',
            'Replace "Your Name" with your real name',
            'Update email, LinkedIn, GitHub links',
            'Customize About section',
            'Update project descriptions',
            'Commit changes'
          ]
        }
      ]
    },
    {
      id: 'project',
      title: '📁 First Project Repository',
      icon: Github,
      color: 'purple',
      steps: [
        {
          id: 1,
          title: 'Create Project Repository',
          description: 'New repository for your Economic Forecasting project',
          details: [
            'Click "+" → New repository',
            'Name: economic-forecasting',
            'Description: Time series forecasting platform',
            'Public, Add README, Python .gitignore',
            'MIT License',
            'Create repository'
          ]
        },
        {
          id: 2,
          title: 'Create Project Structure',
          description: 'Organize your project files',
          details: [
            'Create: data/raw/.gitkeep',
            'Create: data/processed/.gitkeep',
            'Create: notebooks/.gitkeep',
            'Create: src/.gitkeep',
            'Create: models/.gitkeep',
            'Create: results/images/.gitkeep'
          ]
        },
        {
          id: 3,
          title: 'Add Requirements File',
          description: 'List all Python dependencies',
          details: [
            '"Add file" → "Create new file"',
            'Name: requirements.txt',
            'Add: pandas, numpy, matplotlib, etc.',
            'Commit: "Add requirements"'
          ]
        },
        {
          id: 4,
          title: 'Write Professional README',
          description: 'Create comprehensive project documentation',
          details: [
            'Click README.md → Edit',
            'Add project badges at top',
            'Include hero image/screenshot',
            'Write clear overview',
            'Add installation instructions',
            'Include results table',
            'Add tech stack section',
            'Commit changes'
          ]
        },
        {
          id: 5,
          title: 'Upload Project Files',
          description: 'Add your code and notebooks',
          details: [
            'Upload Jupyter notebooks to notebooks/',
            'Upload app.py (Streamlit dashboard)',
            'Upload Python scripts to src/',
            'Upload result images to results/images/',
            'Commit each upload'
          ]
        },
        {
          id: 6,
          title: 'Configure Repository',
          description: 'Set up About section and topics',
          details: [
            'Click Settings',
            'Edit About section (gear icon)',
            'Add description',
            'Add topics: data-science, python, forecasting',
            'Save changes'
          ]
        }
      ]
    },
    {
      id: 'deploy',
      title: '🚀 Deploy Your Project',
      icon: ExternalLink,
      color: 'green',
      steps: [
        {
          id: 1,
          title: 'Sign Up for Streamlit Cloud',
          description: 'Free hosting for your dashboard',
          details: [
            'Go to share.streamlit.io',
            'Click "Sign up"',
            'Connect with GitHub account',
            'Authorize Streamlit'
          ],
          url: 'https://share.streamlit.io'
        },
        {
          id: 2,
          title: 'Deploy Your App',
          description: 'Connect and deploy from GitHub',
          details: [
            'Click "New app"',
            'Select repository: economic-forecasting',
            'Main file: app.py',
            'Click Deploy!',
            'Wait 2-5 minutes'
          ]
        },
        {
          id: 3,
          title: 'Get Your Live URL',
          description: 'Copy your deployed app URL',
          details: [
            'Your app will be at:',
            'https://yourapp.streamlit.app',
            'Copy this URL',
            'Test that it works',
            'Share with others!'
          ]
        },
        {
          id: 4,
          title: 'Update Portfolio Links',
          description: 'Add live demo link to your portfolio',
          details: [
            'Go to yourusername.github.io repo',
            'Edit index.html',
            'Find project card',
            'Update href with Streamlit URL',
            'Commit changes'
          ]
        },
        {
          id: 5,
          title: 'Update Project README',
          description: 'Add live demo link to project',
          details: [
            'Go to economic-forecasting repo',
            'Edit README.md',
            'Update Live Demo link',
            'Add badge for deployment status',
            'Commit changes'
          ]
        },
        {
          id: 6,
          title: 'Test Everything',
          description: 'Verify all links work',
          details: [
            'Visit your portfolio site',
            'Click Live Demo → Should open app',
            'Click Code → Should open repo',
            'Click Article → Should open blog',
            'Test on mobile device'
          ]
        }
      ]
    },
    {
      id: 'blog',
      title: '📝 Create Blog Post',
      icon: FileText,
      color: 'orange',
      steps: [
        {
          id: 1,
          title: 'Create Blog Post File',
          description: 'Add new HTML file in blog folder',
          details: [
            'Go to yourusername.github.io repo',
            'Navigate to blog/ folder',
            '"Add file" → "Create new file"',
            'Name: economic-forecasting.html',
            'Use blog post template'
          ]
        },
        {
          id: 2,
          title: 'Write Your Story',
          description: 'Explain your project journey',
          details: [
            'Start with the problem',
            'Explain your approach',
            'Show key visualizations',
            'Share results and metrics',
            'Discuss challenges faced',
            'Share what you learned'
          ]
        },
        {
          id: 3,
          title: 'Add Code Snippets',
          description: 'Include key technical details',
          details: [
            'Show interesting code examples',
            'Explain your methodology',
            'Include model comparisons',
            'Add syntax highlighting',
            'Keep snippets concise (<20 lines)'
          ]
        },
        {
          id: 4,
          title: 'Add Visualizations',
          description: 'Include charts and graphs',
          details: [
            'Upload images to images/ folder',
            'Reference in blog post',
            'Add descriptive captions',
            'Ensure images are optimized',
            'Use consistent style'
          ]
        },
        {
          id: 5,
          title: 'Link Everything',
          description: 'Cross-reference your work',
          details: [
            'Link to GitHub repository',
            'Link to live demo',
            'Link to portfolio homepage',
            'Add social media links',
            'Include call-to-action'
          ]
        },
        {
          id: 6,
          title: 'Update Blog Index',
          description: 'Add post to blog section on homepage',
          details: [
            'Edit index.html',
            'Find blog section',
            'Add new blog post card',
            'Include title, date, excerpt',
            'Link to blog post',
            'Commit changes'
          ]
        }
      ]
    }
  ];

  const sectionColors = {
    blue: { bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200', icon: 'text-blue-600' },
    purple: { bg: 'bg-purple-50', text: 'text-purple-700', border: 'border-purple-200', icon: 'text-purple-600' },
    green: { bg: 'bg-green-50', text: 'text-green-700', border: 'border-green-200', icon: 'text-green-600' },
    orange: { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', icon: 'text-orange-600' }
  };

  const activeData = sections.find(s => s.id === activeSection);
  const colors = sectionColors[activeData.color];

  const getSectionProgress = (sectionId) => {
    const section = sections.find(s => s.id === sectionId);
    const completed = section.steps.filter(step => 
      completedSteps[`${sectionId}-${step.id}`]
    ).length;
    return Math.round((completed / section.steps.length) * 100);
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8 rounded-lg mb-8 shadow-lg">
        <h1 className="text-4xl font-bold mb-2">🚀 Complete Portfolio Setup Guide</h1>
        <p className="text-xl opacity-90">Step-by-step instructions to build your data science portfolio</p>
        <div className="mt-4 flex gap-4">
          <div className="bg-white bg-opacity-20 px-4 py-2 rounded">
            <div className="text-2xl font-bold">4</div>
            <div className="text-sm">Major Steps</div>
          </div>
          <div className="bg-white bg-opacity-20 px-4 py-2 rounded">
            <div className="text-2xl font-bold">24</div>
            <div className="text-sm">Total Tasks</div>
          </div>
          <div className="bg-white bg-opacity-20 px-4 py-2 rounded">
            <div className="text-2xl font-bold">~4hr</div>
            <div className="text-sm">Total Time</div>
          </div>
        </div>
      </div>

      {/* Section Tabs */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        {sections.map(section => {
          const Icon = section.icon;
          const progress = getSectionProgress(section.id);
          const isActive = activeSection === section.id;
          
          return (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`p-4 rounded-lg border-2 transition-all ${
                isActive
                  ? 'bg-white border-blue-500 shadow-lg scale-105'
                  : 'bg-white border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="flex items-center gap-3 mb-2">
                <Icon className={isActive ? 'text-blue-600' : 'text-gray-400'} size={24} />
                <h3 className={`font-semibold text-left ${isActive ? 'text-blue-900' : 'text-gray-700'}`}>
                  {section.title}
                </h3>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-gray-600 mt-2">{progress}% Complete</p>
            </button>
          );
        })}
      </div>

      {/* Active Section Content */}
      <div className={`${colors.bg} ${colors.border} border-2 rounded-lg p-8`}>
        <div className="flex items-center gap-3 mb-6">
          {React.createElement(activeData.icon, { 
            className: `${colors.icon}`, 
            size: 32 
          })}
          <h2 className={`text-3xl font-bold ${colors.text}`}>{activeData.title}</h2>
        </div>

        {/* Steps */}
        <div className="space-y-4">
          {activeData.steps.map((step, idx) => {
            const isCompleted = completedSteps[`${activeData.id}-${step.id}`];
            
            return (
              <div key={step.id} className="bg-white rounded-lg shadow-sm border border-gray-200">
                {/* Step Header */}
                <div
                  onClick={() => toggleStep(activeData.id, step.id)}
                  className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start gap-4">
                    {/* Checkbox */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleStep(activeData.id, step.id);
                      }}
                      className="flex-shrink-0 mt-1"
                    >
                      {isCompleted ? (
                        <CheckCircle className="text-green-600" size={24} />
                      ) : (
                        <Circle className="text-gray-400" size={24} />
                      )}
                    </button>

                    {/* Content */}
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={`text-sm font-bold ${colors.text} bg-white px-2 py-1 rounded`}>
                          Step {idx + 1}
                        </span>
                        <h3 className={`text-lg font-bold ${isCompleted ? 'text-gray-500 line-through' : 'text-gray-900'}`}>
                          {step.title}
                        </h3>
                      </div>
                      <p className="text-gray-600">{step.description}</p>
                      
                      {step.url && (
                        <a
                          href={step.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-2 mt-2 text-blue-600 hover:text-blue-800 text-sm"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <ExternalLink size={16} />
                          Open Link
                        </a>
                      )}
                    </div>

                    {/* Expand Arrow */}
                    <ChevronRight 
                      className={`text-gray-400 transition-transform ${isCompleted ? 'rotate-90' : ''}`}
                      size={20}
                    />
                  </div>
                </div>

                {/* Step Details (Expanded when completed) */}
                {isCompleted && (
                  <div className="px-4 pb-4 pl-16 border-t border-gray-100 pt-4">
                    <h4 className="font-semibold text-gray-900 mb-2">Detailed Steps:</h4>
                    <ol className="space-y-2">
                      {step.details.map((detail, detailIdx) => (
                        <li key={detailIdx} className="flex items-start gap-2">
                          <span className="text-blue-600 font-bold text-sm mt-0.5">
                            {detailIdx + 1}.
                          </span>
                          <span className="text-gray-700">{detail}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Section Summary */}
        <div className={`mt-8 p-6 ${colors.bg} border-2 ${colors.border} rounded-lg`}>
          <h3 className={`text-xl font-bold ${colors.text} mb-3`}>
            {getSectionProgress(activeData.id) === 100 ? '🎉 Section Complete!' : '📋 Section Summary'}
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Progress</p>
              <p className={`text-2xl font-bold ${colors.text}`}>
                {getSectionProgress(activeData.id)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Steps Completed</p>
              <p className={`text-2xl font-bold ${colors.text}`}>
                {activeData.steps.filter(s => completedSteps[`${activeData.id}-${s.id}`]).length} / {activeData.steps.length}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Links */}
      <div className="mt-8 bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-xl font-bold text-gray-900 mb-4">🔗 Quick Links</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <Github size={24} />
            <div>
              <div className="font-semibold">GitHub</div>
              <div className="text-sm text-gray-600">Create account & repos</div>
            </div>
          </a>
          <a
            href="https://share.streamlit.io"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <ExternalLink size={24} />
            <div>
              <div className="font-semibold">Streamlit Cloud</div>
              <div className="text-sm text-gray-600">Deploy your apps</div>
            </div>
          </a>
          <a
            href="https://fred.stlouisfed.org/docs/api/api_key.html"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
          >
            <FileText size={24} />
            <div>
              <div className="font-semibold">FRED API Key</div>
              <div className="text-sm text-gray-600">Get data access</div>
            </div>
          </a>
        </div>
      </div>

      {/* Tips */}
      <div className="mt-8 bg-yellow-50 border-2 border-yellow-200 p-6 rounded-lg">
        <h3 className="text-xl font-bold text-yellow-900 mb-3">💡 Pro Tips</h3>
        <ul className="space-y-2 text-yellow-800">
          <li>• <strong>Save your work frequently:</strong> Commit changes after each step</li>
          <li>• <strong>Use descriptive names:</strong> Clear file and folder names help organization</li>
          <li>• <strong>Test everything:</strong> Click all links to verify they work</li>
          <li>• <strong>Mobile check:</strong> View your portfolio on phone/tablet</li>
          <li>• <strong>Ask for help:</strong> GitHub has great documentation if you get stuck</li>
        </ul>
      </div>
    </div>
  );
};

export default SetupGuide;