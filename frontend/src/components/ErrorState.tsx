import React from 'react';
import { FiAlertTriangle } from 'react-icons/fi';

interface ErrorStateProps {
  message: string;
}

export function ErrorState({ message }: ErrorStateProps) {
  return (
    <div className="flex items-start sm:items-center gap-2 sm:gap-3 rounded-2xl sm:rounded-3xl border border-rose-500/40 bg-rose-500/10 px-3 py-3 sm:px-5 sm:py-4 text-rose-200">
      <FiAlertTriangle size={18} className="sm:w-5 sm:h-5 flex-shrink-0 mt-0.5 sm:mt-0" />
      <p className="text-xs sm:text-sm break-words">{message}</p>
    </div>
  );
}

